#!/usr/bin/env python3
"""
eval_val.py — Evaluate GPT-OSS 20B on the validation set.

Pipeline:
  1. Parse val.txt  →  extract question_id, problem, gold_answer (integer)
  2. Run vLLM inference with GPT-OSS 20B  →  k candidate outputs per question
  3. For each candidate:
       • Extract final integer answer  (regex / \boxed{})
       • Exact-match vs gold (if available)
       • Reasoning verdict  (Claude API if key set, else heuristic)
  4. Compute all aggregate & summary metrics per the evaluation schema
  5. Save results to eval_results.json

Usage:
  # With Claude evaluation (best quality):
  ANTHROPIC_API_KEY=sk-ant-... python eval_val.py

  # Without API key (heuristic reasoning verdict):
  python eval_val.py

  # Override paths:
  python eval_val.py --model ~/model --val ~/data/aimo_cleaned_data_v1/val.txt \
                     --output ~/eval_results.json --k 4
"""

import os, re, json, argparse, textwrap
from collections import Counter
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = os.path.expanduser("~/model")
DEFAULT_VAL    = os.path.expanduser("~/data/aimo_cleaned_data_v1/val.txt")
DEFAULT_OUT    = os.path.expanduser("~/eval_results.json")
DEFAULT_K      = 4          # candidates per question
DEFAULT_TEMP   = 0.7
DEFAULT_TOKENS = 2048
SEPARATOR      = "=" * 50

EVAL_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast for eval

# ── 1. Parse val.txt ──────────────────────────────────────────────────────────

def _extract_int_from_boxed(text: str):
    """Return integer string from \\boxed{N} or None."""
    if not text:
        return None
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return None
    inner = m.group(1).replace(",", "").strip()
    # Accept only plain integers (no fractions, letters, etc.)
    if re.fullmatch(r"-?\d+", inner):
        return inner
    return None


def parse_val(val_file: str) -> list[dict]:
    with open(val_file) as f:
        content = f.read()
    entries = [e.strip() for e in content.split(SEPARATOR) if e.strip()]
    questions = []
    for entry in entries:
        qid_m = re.search(r"<question_id>(.*?)</question_id>", entry, re.DOTALL)
        q_m   = re.search(r"<question>(.*?)</question>",       entry, re.DOTALL)
        fa_m  = re.search(r"<final_answer>(.*?)</final_answer>",entry, re.DOTALL)
        if not qid_m or not q_m:
            continue
        raw_fa   = fa_m.group(1).strip() if fa_m else ""
        gold_int = _extract_int_from_boxed(raw_fa)   # None if not an integer
        questions.append({
            "question_id": qid_m.group(1).strip(),
            "problem":     q_m.group(1).strip(),
            "gold_answer": gold_int,               # str integer or None
        })
    return questions


# ── 2. vLLM inference ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert mathematician solving olympiad competition problems. "
    "Think step by step. At the end of your solution write your final integer "
    "answer inside \\boxed{}, for example: \\boxed{42}."
)

def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Problem:\n{problem}"},
    ]
    # Use the model's own chat template for correct formatting
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_inference(questions: list[dict], model_path: str,
                  k: int, max_tokens: int, temperature: float) -> list[list[dict]]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_path}…")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}…")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
    )

    sampling = SamplingParams(
        n=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    prompts = [build_prompt(q["problem"], tokenizer) for q in questions]
    print(f"Running inference: {len(prompts)} questions × k={k}…")
    outputs = llm.generate(prompts, sampling)

    results = []
    for output in outputs:
        candidates = [
            {"candidate_id": f"c{j+1}", "text": o.text}
            for j, o in enumerate(output.outputs)
        ]
        results.append(candidates)
    return results


# ── 3. Answer extraction ──────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """
    Return the final integer answer as a string, or 'NONE'.
    Priority: last \\boxed{} > "the answer is N" patterns > last standalone integer.
    """
    # (a) All \boxed{} occurrences — take the last one
    boxed_all = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed_all:
        inner = boxed_all[-1].replace(",", "").strip()
        if re.fullmatch(r"-?\d+", inner):
            return inner

    # (b) "answer is / equals / = N" near the end of the text
    tail = text[-500:]  # only look in last ~500 chars
    patterns = [
        r"(?:answer|result|value)\s+(?:is|=|equals)\s*[:\s]*(-?\d{1,9})\b",
        r"(?:therefore|thus|hence|so)[^.]*?=\s*(-?\d{1,9})\b",
        r"=\s*\**(-?\d{1,9})\**\s*[.$\n]",
    ]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return m.group(1)

    # (c) Last standalone integer on its own line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"-?\d{1,9}", line):
            return line

    return "NONE"


# ── 4. Heuristic reasoning verdict ───────────────────────────────────────────

def heuristic_verdict(text: str, extracted: str, gold: str | None) -> tuple[str, str, int, str]:
    """
    Returns (reasoning_verdict, error_type, score, short_reason).
    Used when Claude API is unavailable.
    """
    has_steps    = len(re.findall(r"\n", text)) >= 3
    has_equations = bool(re.search(r"[=+\-*/^]", text))
    word_count   = len(text.split())
    is_short     = word_count < 40

    if extracted == "NONE":
        return "incorrect", "format", 1, "No integer answer found in output."

    if gold is not None:
        exact = extracted == gold
        if exact:
            if is_short or not has_steps:
                return "suspicious", "unsupported_final_answer", 6, \
                       "Correct answer but reasoning is too brief to verify."
            return "correct", "none", 9, \
                   "Answer matches gold; reasoning shows clear steps."
        else:
            # Wrong answer — look for common error types
            if not has_equations:
                err = "logic"
            elif re.search(r"\d+\s*[+\-*/]\s*\d+", text):
                err = "arithmetic"
            else:
                err = "algebra"
            return "incorrect", err, 2, \
                   f"Extracted {extracted} but gold is {gold}."
    else:
        # No gold — judge by presence of coherent reasoning only
        if is_short or not has_steps:
            return "suspicious", "unsupported_final_answer", 5, \
                   "No gold answer; reasoning appears incomplete."
        return "suspicious", "none", 6, \
               "No gold answer available; reasoning looks structured but unverified."


# ── 5. Claude-based evaluation ────────────────────────────────────────────────

EVAL_SYSTEM = textwrap.dedent("""\
    You are a strict evaluator for olympiad-style math problems with integer final answers.

    Evaluation rules:
    1. Extract the model's final integer answer from each candidate output.
    2. If gold_answer is provided, compare it with the extracted answer.
    3. If gold_answer is not provided, set exact_match to null.
    4. Judge whether the reasoning is mathematically valid.
    5. If the final answer is correct but the reasoning is weak or unsupported, mark it "suspicious".
    6. If no integer final answer can be extracted, set extracted_answer to "NONE".
    7. Focus on mathematical correctness, not writing style.
    8. Be strict and deterministic. Do not use outside knowledge.
    9. Evaluate each candidate independently.
    10. Select the best candidate (mathematical correctness > reasoning validity > score).

    For each candidate return:
      candidate_id, extracted_answer (integer string or "NONE"),
      exact_match (0/1 or null), format_valid (0/1),
      reasoning_verdict ("correct"|"incorrect"|"suspicious"),
      error_type ("none"|"arithmetic"|"algebra"|"logic"|"missing_case"|
                  "unsupported_final_answer"|"format"|"other"),
      score (0-10), short_reason (one sentence).

    Return ONLY valid JSON. No markdown fences. Schema:
    {"candidates": [{...}, ...]}
""")


def claude_eval(question_id, problem, gold, candidates, api_key):
    import anthropic

    lines = [
        f"question_id: {question_id}",
        f"problem: {problem}",
    ]
    if gold:
        lines.append(f"gold_answer: {gold}")
    lines.append("")
    for c in candidates:
        lines.append(f"--- {c['candidate_id']} ---")
        lines.append(c["text"].strip())
        lines.append("")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=4096,
        system=EVAL_SYSTEM,
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)["candidates"]


# ── 6. Evaluate one question ──────────────────────────────────────────────────

def evaluate_question(q: dict, candidates: list[dict], api_key: str | None) -> list[dict]:
    if api_key:
        try:
            return claude_eval(
                q["question_id"], q["problem"], q["gold_answer"],
                candidates, api_key
            )
        except Exception as e:
            print(f"  [warn] Claude eval failed ({e}), using heuristic fallback.")

    # Heuristic path
    evals = []
    for c in candidates:
        extracted = extract_answer(c["text"])
        fmt_valid = 0 if extracted == "NONE" else 1
        exact = None
        if q["gold_answer"] is not None:
            exact = 1 if extracted == q["gold_answer"] else 0
        verdict, err, score, reason = heuristic_verdict(
            c["text"], extracted, q["gold_answer"]
        )
        evals.append({
            "candidate_id":      c["candidate_id"],
            "extracted_answer":  extracted,
            "exact_match":       exact,
            "format_valid":      fmt_valid,
            "reasoning_verdict": verdict,
            "error_type":        err,
            "score":             score,
            "short_reason":      reason,
        })
    return evals


# ── 7. Aggregate metrics ──────────────────────────────────────────────────────

VERDICT_RANK = {"correct": 0, "suspicious": 1, "incorrect": 2}


def compute_aggregate(gold: str | None, evals: list[dict]) -> dict:
    has_gold = gold is not None
    n        = len(evals)
    extracted = [c["extracted_answer"] for c in evals]
    exact_ok  = [c for c in evals if c.get("exact_match") == 1]

    # majority vote (exclude NONE)
    non_none = [a for a in extracted if a != "NONE"]
    if non_none:
        ctr = Counter(non_none)
        maj_ans, maj_cnt = ctr.most_common(1)[0]
    else:
        maj_ans, maj_cnt = "NONE", extracted.count("NONE")
    maj_frac = round(maj_cnt / n, 4)

    # best candidate
    best = sorted(evals, key=lambda c: (VERDICT_RANK.get(c["reasoning_verdict"], 2), -c["score"]))[0]

    scores        = [c["score"] for c in evals]
    correct_scores = [c["score"] for c in exact_ok]
    n_exact        = len(exact_ok)

    return {
        "best_candidate_id":          best["candidate_id"],
        "best_extracted_answer":      best["extracted_answer"],
        "best_of_k_exact_match":      best.get("exact_match") if has_gold else None,
        "pass_at_k":                  (1 if n_exact > 0 else 0) if has_gold else None,
        "num_candidates":             n,
        "num_exact_match":            n_exact            if has_gold else None,
        "exact_match_rate_at_k":      round(n_exact / n, 4) if has_gold else None,
        "majority_answer":            maj_ans,
        "majority_answer_count":      maj_cnt,
        "majority_answer_fraction":   maj_frac,
        "majority_vote_exact_match":  (1 if maj_ans == gold else 0) if has_gold else None,
        "num_unique_extracted_answers": len(set(extracted)),
        "unique_answer_list":         list(set(extracted)),
        "answer_histogram":           dict(Counter(extracted)),
        "num_format_valid":           sum(c["format_valid"] for c in evals),
        "format_valid_rate_at_k":     round(sum(c["format_valid"] for c in evals) / n, 4),
        "num_none_answers":           extracted.count("NONE"),
        "num_reasoning_correct":      sum(1 for c in evals if c["reasoning_verdict"] == "correct"),
        "num_reasoning_suspicious":   sum(1 for c in evals if c["reasoning_verdict"] == "suspicious"),
        "num_reasoning_incorrect":    sum(1 for c in evals if c["reasoning_verdict"] == "incorrect"),
        "reasoning_correct_rate_at_k": round(
            sum(1 for c in evals if c["reasoning_verdict"] == "correct") / n, 4),
        "avg_score":                  round(sum(scores) / n, 2),
        "max_score":                  max(scores),
        "correct_answer_support_count": n_exact if has_gold else None,
        "correct_answer_best_score":    max(correct_scores)                       if correct_scores else None,
        "correct_answer_avg_score":     round(sum(correct_scores)/len(correct_scores), 2) if correct_scores else None,
    }


# ── 8. Summary ────────────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    s = {
        "num_questions":                  len(results),
        "num_questions_with_gold_answer": sum(1 for r in results if r["problem_has_gold_answer"]),
        "num_total_candidates":           sum(len(r["candidates"]) for r in results),
        "num_exact_match":                0,
        "num_format_valid":               0,
        "num_reasoning_correct":          0,
        "num_reasoning_suspicious":       0,
        "num_pass_at_k":                  0,
        "num_best_of_k_exact_match":      0,
        "num_majority_vote_exact_match":  0,
    }
    for r in results:
        agg = r["aggregate"]
        s["num_format_valid"]       += agg["num_format_valid"]
        s["num_reasoning_correct"]  += agg["num_reasoning_correct"]
        s["num_reasoning_suspicious"] += agg["num_reasoning_suspicious"]
        if r["problem_has_gold_answer"]:
            s["num_exact_match"]             += agg["num_exact_match"] or 0
            s["num_pass_at_k"]               += agg["pass_at_k"] or 0
            s["num_best_of_k_exact_match"]   += agg["best_of_k_exact_match"] or 0
            s["num_majority_vote_exact_match"] += agg["majority_vote_exact_match"] or 0
    return s


# ── 9. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS 20B on val set")
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help="Path to model")
    parser.add_argument("--val",    default=DEFAULT_VAL,    help="Path to val.txt")
    parser.add_argument("--output", default=DEFAULT_OUT,    help="Output JSON path")
    parser.add_argument("--k",      type=int, default=DEFAULT_K, help="Candidates per question")
    parser.add_argument("--temp",   type=float, default=DEFAULT_TEMP)
    parser.add_argument("--tokens", type=int,   default=DEFAULT_TOKENS)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    if api_key:
        print(f"Claude eval: enabled  (model={EVAL_MODEL})")
    else:
        print("Claude eval: disabled (set ANTHROPIC_API_KEY to enable) — using heuristics")

    # Step 1: parse
    questions = parse_val(args.val)
    print(f"Loaded {len(questions)} validation questions")
    gold_count = sum(1 for q in questions if q["gold_answer"] is not None)
    print(f"  {gold_count} have integer gold answers")

    # Step 2: inference
    all_candidates = run_inference(
        questions, args.model, args.k, args.tokens, args.temp
    )

    # Step 3-4: evaluate + aggregate
    results = []
    for i, (q, candidates) in enumerate(zip(questions, all_candidates)):
        print(f"[{i+1}/{len(questions)}] Evaluating {q['question_id']}  gold={q['gold_answer']}")
        evals     = evaluate_question(q, candidates, api_key)
        aggregate = compute_aggregate(q["gold_answer"], evals)
        results.append({
            "question_id":           q["question_id"],
            "problem_has_gold_answer": q["gold_answer"] is not None,
            "candidates":            evals,
            "aggregate":             aggregate,
        })
        # Print quick progress
        print(f"         best={aggregate['best_extracted_answer']}  "
              f"pass@k={aggregate['pass_at_k']}  "
              f"avg_score={aggregate['avg_score']}")

    # Step 5: summary
    summary = compute_summary(results)

    output = {"results": results, "summary": summary}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Done. Saved to {args.output} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
