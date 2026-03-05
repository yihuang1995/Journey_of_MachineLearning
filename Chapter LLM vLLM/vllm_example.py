"""
vLLM Example: Offline Batch Inference & OpenAI-Compatible Server
================================================================
vLLM is a fast and memory-efficient inference engine for LLMs.
Key features:
  - PagedAttention: manages KV-cache in pages (like virtual memory) to eliminate memory waste
  - Continuous batching: dynamically adds new requests into running batches
  - Optimized CUDA kernels: faster attention computation

How to install:
  pip install vllm

How to run:
  python vllm_example.py
"""

# ──────────────────────────────────────────────
# SECTION 1: Offline Batch Inference
# ──────────────────────────────────────────────
from vllm import LLM, SamplingParams

# ── 1a. Load the model ──────────────────────────────────────────────────────
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    # ^ HuggingFace model name or local path.
    #   vLLM downloads and caches the weights automatically.

    tokenizer=None,
    # ^ Custom tokenizer name/path. None = use the model's built-in tokenizer.

    tokenizer_mode="auto",
    # ^ "auto"  → use fast HF tokenizer if available, else fall back to slow.
    #   "slow"  → always use the Python tokenizer (useful for debugging).

    trust_remote_code=False,
    # ^ Set True only if the model repo includes custom Python code that must run
    #   during loading (e.g., some Falcon / Qwen variants).
    #   Leave False for safety with untrusted repos.

    dtype="auto",
    # ^ Weight / computation dtype.
    #   "auto"    → picks bfloat16 on Ampere+ GPUs, float16 otherwise.
    #   "float16" → half precision (good for most consumer GPUs).
    #   "bfloat16"→ better numerical stability for training-style use.
    #   "float32" → full precision (slow, large memory footprint).

    quantization=None,
    # ^ Post-training quantization format.
    #   None       → no quantization (full dtype above).
    #   "awq"      → Activation-aware Weight Quantization (4-bit).
    #   "gptq"     → GPTQ 4-bit quantization.
    #   "squeezellm" → SqueezeLLM sparse-quantization.

    revision=None,
    # ^ A specific git commit hash or branch of the HuggingFace repo to pin to.
    #   Useful for reproducibility. None = latest main branch.

    tokenizer_revision=None,
    # ^ Same as `revision` but only for the tokenizer files.

    seed=42,
    # ^ Random seed for reproducible sampling across runs.

    gpu_memory_utilization=0.90,
    # ^ Fraction of GPU VRAM vLLM is allowed to use (0.0 – 1.0).
    #   0.90 → use 90 % of available GPU memory for the KV-cache + weights.
    #   Lower this if you share the GPU with other processes.

    swap_space=4,
    # ^ CPU RAM (in GiB) used as a swap space for KV-cache blocks that don't
    #   fit in GPU memory. Increases max concurrent sequences at a latency cost.

    max_model_len=4096,
    # ^ Maximum number of tokens (prompt + generated) the model handles.
    #   Overrides the model's default context length.
    #   Larger = more memory for KV-cache.

    tensor_parallel_size=1,
    # ^ Number of GPUs for tensor parallelism (splits weight matrices across GPUs).
    #   1 = single GPU. Set to 2/4/8 for multi-GPU inference.
    #   Requires NCCL; all GPUs must be on the same node.

    pipeline_parallel_size=1,
    # ^ Number of pipeline stages (splits layers across GPUs sequentially).
    #   Combine with tensor_parallel_size for very large models.
    #   pipeline_parallel_size=2, tensor_parallel_size=4 → uses 8 GPUs total.

    max_num_batched_tokens=None,
    # ^ Hard cap on total tokens (across all sequences) processed per forward pass.
    #   None = vLLM chooses automatically based on available memory.

    max_num_seqs=256,
    # ^ Maximum number of sequences processed concurrently in one scheduler step.

    enforce_eager=False,
    # ^ False → use CUDA graphs for faster inference (recommended).
    #   True  → disable CUDA graphs; useful for debugging or unsupported ops.

    enable_prefix_caching=False,
    # ^ Cache KV states for shared prompt prefixes across requests.
    #   Great for chat systems where every message starts with the same system prompt.
)


# ── 1b. Define sampling parameters ──────────────────────────────────────────
sampling_params = SamplingParams(
    n=1,
    # ^ Number of output sequences to generate per prompt.
    #   n=3 → returns 3 different completions for the same prompt (useful for best-of-N).

    best_of=1,
    # ^ Pool size for beam/sampling. vLLM generates `best_of` candidates internally
    #   and returns the top `n`. Must be >= n.
    #   best_of=5, n=1 → generate 5 candidates, return the highest-scoring one.

    presence_penalty=0.0,
    # ^ Penalises tokens that have appeared at least once in the output so far.
    #   Range: –2.0 (encourage repetition) to 2.0 (discourage repetition).
    #   0.0 = no penalty.

    frequency_penalty=0.0,
    # ^ Penalises tokens proportional to how often they've appeared.
    #   Range: –2.0 to 2.0. Stronger than presence_penalty for repeat offenders.

    repetition_penalty=1.0,
    # ^ Multiplicative penalty on already-seen tokens (from HuggingFace Transformers).
    #   1.0 = no effect. >1.0 discourages repeats. <1.0 encourages them.

    temperature=0.7,
    # ^ Controls randomness of the output distribution.
    #   0.0       → greedy decoding (always pick the highest-prob token). Deterministic.
    #   0.1–0.4   → focused, factual responses.
    #   0.7–0.9   → balanced creativity and coherence.
    #   1.0       → raw model probabilities.
    #   >1.0      → more random / chaotic output.

    top_p=0.9,
    # ^ Nucleus sampling: keep only the smallest set of tokens whose cumulative
    #   probability ≥ top_p, then sample from that set.
    #   1.0 = consider all tokens. 0.9 = cut off the long tail.
    #   Works in tandem with temperature. Disable with 1.0 if using top_k.

    top_k=50,
    # ^ Sample only from the top-k highest probability tokens.
    #   -1 = disabled (consider all tokens).
    #   Typical values: 20–100. Lower = more focused.

    min_p=0.0,
    # ^ Minimum probability threshold (relative to the top token) for a token to
    #   be considered. Filters very unlikely tokens without a hard k cutoff.
    #   0.0 = disabled.

    use_beam_search=False,
    # ^ True = beam search (deterministic, higher quality for translation/summarisation).
    # ^ False = sampling (diverse, creative output). Requires temperature > 0.

    length_penalty=1.0,
    # ^ Applied only when use_beam_search=True.
    #   >1.0 favours longer sequences; <1.0 favours shorter ones.

    early_stopping=False,
    # ^ Beam search only. True = stop when all beams hit EOS.
    #   "never" = always generate max_tokens even after EOS.

    stop=["</s>", "\n\nHuman:"],
    # ^ List of strings that immediately terminate generation when produced.
    #   The stop string itself is NOT included in the output.
    #   Useful for chat templates or structured outputs.

    stop_token_ids=[],
    # ^ Same as `stop` but specified as token IDs (integers).

    max_tokens=512,
    # ^ Maximum number of NEW tokens to generate (does not include prompt tokens).
    #   Generation stops at max_tokens OR a stop condition, whichever comes first.

    min_tokens=0,
    # ^ Force the model to generate at least this many tokens before allowing EOS.
    #   Prevents trivially short outputs for open-ended prompts.

    logprobs=None,
    # ^ If set to an integer k, return the log-probabilities of the top-k tokens
    #   at each position. Useful for scoring or uncertainty estimation.
    #   None = don't return logprobs.

    prompt_logprobs=None,
    # ^ Same as logprobs but for the prompt tokens (i.e., compute the model's
    #   perplexity on the given input). None = disabled.

    skip_special_tokens=True,
    # ^ True  = strip special tokens (e.g., <s>, </s>, <pad>) from the output text.
    # ^ False = keep them (useful when debugging tokenization).

    spaces_between_special_tokens=True,
    # ^ Add a space between consecutive special tokens in the decoded string.

    logits_processors=None,
    # ^ A list of callables that modify the raw logit tensor before sampling.
    #   Use to implement custom constraints, biases, or grammar-based decoding.
)


# ── 1c. Run inference ────────────────────────────────────────────────────────
prompts = [
    "Explain the attention mechanism in transformers in simple terms.",
    "What is PagedAttention and why does vLLM use it?",
    "Write a haiku about GPU memory.",
]

outputs = llm.generate(
    prompts,
    # ^ A list of raw strings OR a list of token-id lists.

    sampling_params=sampling_params,
    # ^ The SamplingParams object defined above.

    use_tqdm=True,
    # ^ Show a progress bar while processing the batch.
)

print("\n" + "=" * 60)
print("OFFLINE BATCH INFERENCE RESULTS")
print("=" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # output.outputs is a list of CompletionOutput objects (one per n).
    # Each has: .text (str), .token_ids (list[int]), .logprobs, .finish_reason

    print(f"\nPrompt : {prompt!r}")
    print(f"Output : {generated_text!r}")
    print(f"Finish : {output.outputs[0].finish_reason}")
    # finish_reason: "stop" | "length" | "abort"
    # "stop"   → hit a stop string or EOS token
    # "length" → hit max_tokens limit
    # "abort"  → request was cancelled


# ──────────────────────────────────────────────
# SECTION 2: OpenAI-Compatible API Server
# ──────────────────────────────────────────────
"""
vLLM ships a drop-in OpenAI-compatible REST server.
Start it from the terminal (no Python code needed):

    vllm serve meta-llama/Llama-3.2-1B-Instruct \
        --host 0.0.0.0 \
        --port 8000 \
        --dtype auto \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        --tensor-parallel-size 1 \
        --api-key "my-secret-key"   # optional, enables bearer-token auth

Then query it with the standard OpenAI Python client:
"""

from openai import OpenAI

def query_vllm_server():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        # ^ Point to your vLLM server instead of api.openai.com

        api_key="my-secret-key",
        # ^ Must match --api-key on the server; any non-empty string if auth is off.
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        # ^ Must exactly match the model name used when starting the server.

        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user",   "content": "What is vLLM?"},
        ],

        temperature=0.7,       # same semantics as SamplingParams above
        top_p=0.9,
        max_tokens=256,
        n=1,                   # number of completions
        stop=None,             # stop strings
        stream=False,          # True = server-sent events for token streaming
    )

    print("\n" + "=" * 60)
    print("OPENAI-COMPATIBLE SERVER RESULT")
    print("=" * 60)
    print(response.choices[0].message.content)
    print(f"Finish reason : {response.choices[0].finish_reason}")
    print(f"Prompt tokens : {response.usage.prompt_tokens}")
    print(f"Output tokens : {response.usage.completion_tokens}")


# Uncomment to test against a running vLLM server:
# query_vllm_server()


# ──────────────────────────────────────────────
# SECTION 3: Streaming Tokens (Offline API)
# ──────────────────────────────────────────────
def streaming_example():
    """Generate tokens and print them as they arrive."""
    sampling_params_stream = SamplingParams(temperature=0.8, max_tokens=200)

    # llm.generate() is batched & blocking. For true streaming use AsyncLLMEngine:
    #   from vllm import AsyncLLMEngine, AsyncEngineArgs
    #   engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model="..."))
    #   async for output in engine.generate(prompt, sampling_params, request_id="0"):
    #       print(output.outputs[0].text, end="", flush=True)

    # Simple simulation with the blocking API:
    result = llm.generate(["Tell me a short story about a robot."], sampling_params_stream)
    print("\nStreamed output (simulated):")
    for token in result[0].outputs[0].text.split():
        print(token, end=" ", flush=True)
    print()


streaming_example()
