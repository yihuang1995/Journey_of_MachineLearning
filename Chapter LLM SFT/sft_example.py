"""
Supervised Fine-Tuning (SFT) with Unsloth + TRL
=================================================
Stack used (same as the AIMO-3 Kaggle environment):
  - unsloth  : memory-efficient 4-bit QLoRA loading + patched attention kernels
  - trl      : SFTTrainer (wraps HuggingFace Trainer for instruction-tuning)
  - peft     : LoRA adapter config
  - datasets : dataset loading / formatting

Install:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl peft datasets accelerate bitsandbytes

Goal of this file:
  Fine-tune GPT-OSS 120B (the model used in the AIMO-3 Kaggle notebook) on a
  math QA dataset so it learns to output answers inside \\boxed{}.

  Model path (Kaggle): /kaggle/input/gpt-oss-120b/transformers/default/1
  GPT-OSS 120B requires at least 2× A100 80GB (tensor_parallel_size=2) for
  4-bit QLoRA, or 4–8× for full bf16.
"""

# ──────────────────────────────────────────────
# SECTION 1: Load Model + Tokenizer (QLoRA / 4-bit)
# ──────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch

MODEL_PATH = "/kaggle/input/gpt-oss-120b/transformers/default/1"
# ^ Local path to GPT-OSS 120B weights (Kaggle competition environment).
#   Outside Kaggle, replace with any HuggingFace repo or local directory.

MAX_SEQ_LEN = 4096
# ^ Maximum token length for both prompt and completion combined.
#   GPT-OSS 120B supports long contexts; 4096 is a practical SFT default.
#   Increase to 8192–65536 for problems that need long chain-of-thought.

DTYPE = None
# ^ None = auto-detect: bfloat16 on Ampere+ GPUs, float16 otherwise.
#   You can also force torch.float16 or torch.bfloat16.

LOAD_IN_4BIT = True
# ^ True  = load weights in 4-bit (QLoRA). Cuts VRAM ~4x vs bfloat16.
#            GPT-OSS 120B in bf16 needs ~240 GB; in 4-bit QLoRA ~60–70 GB,
#            fitting across 2× A100 80GB with tensor_parallel_size=2.
# ^ False = full precision LoRA — needs 4–8× A100 for a 120B model.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    # ^ Local path or HuggingFace repo for GPT-OSS 120B.
    #   Unsloth will shard the weights across all visible GPUs automatically.

    max_seq_length=MAX_SEQ_LEN,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,

    token=None,
    # ^ HuggingFace access token — not needed for local Kaggle paths.
    #   Required when loading gated HF repos (e.g. Llama-3, Gemma).
)


# ──────────────────────────────────────────────
# SECTION 2: Attach LoRA Adapters (PEFT)
# ──────────────────────────────────────────────
# LoRA (Low-Rank Adaptation) freezes the original weights and injects two small
# trainable matrices A (d×r) and B (r×d) into selected layers.
# Only A and B are trained — the original model is untouched.

model = FastLanguageModel.get_peft_model(
    model,

    r=16,
    # ^ LoRA rank. Controls the size of the adapter matrices.
    #   Higher r = more expressive adapters but more parameters & VRAM.
    #   Common choices: 8, 16, 32, 64.
    #   Rule of thumb: start with 16; increase if underfitting.

    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
        "gate_proj", "up_proj", "down_proj",        # MLP (FFN) projections
    ],
    # ^ Which weight matrices to inject LoRA into.
    #   Attention + MLP = full coverage. Attention-only = lighter but weaker.

    lora_alpha=32,
    # ^ Scaling factor for the LoRA update: effective_lr *= lora_alpha / r.
    #   Set to 2× r as a baseline (here: 32 = 2 × 16).
    #   Higher alpha = larger effective learning rate for the adapters.

    lora_dropout=0.05,
    # ^ Dropout applied to LoRA layers during training.
    #   0.0 = no dropout (fine for small datasets; may overfit on large ones).
    #   0.05–0.10 = slight regularization.

    bias="none",
    # ^ Whether to train bias terms alongside LoRA.
    #   "none"  = don't train biases (most memory-efficient, usually sufficient).
    #   "all"   = train all biases.
    #   "lora_only" = train only biases in LoRA layers.

    use_gradient_checkpointing="unsloth",
    # ^ Recomputes activations during backward pass instead of storing them.
    #   "unsloth" = unsloth's optimized checkpointing (30% less VRAM than HF default).
    #   True      = standard HF gradient checkpointing.
    #   False     = store all activations (fastest training, most VRAM).

    random_state=42,
    # ^ Seed for reproducible LoRA weight initialization.

    use_rslora=False,
    # ^ Rank-Stabilized LoRA: normalizes adapters so effective LR is independent of r.
    #   Recommended when sweeping over different r values.

    loftq_config=None,
    # ^ LoftQ initialization: quantize base weights and initialize LoRA to compensate.
    #   Improves accuracy when LOAD_IN_4BIT=True. Set to LoftQConfig() to enable.
)


# ──────────────────────────────────────────────
# SECTION 3: Dataset Preparation
# ──────────────────────────────────────────────
from datasets import Dataset

# Each training example is a dict with "prompt" and "completion".
# The model learns to generate "completion" given "prompt".

RAW_DATA = [
    {
        "problem": "What is the sum of all integers from 1 to 100?",
        "solution": (
            "The sum of consecutive integers 1 to n is n(n+1)/2.\n"
            "For n=100: 100 × 101 / 2 = 5050.\n"
            "\\boxed{5050}"
        ),
    },
    {
        "problem": "Find the number of prime numbers less than 20.",
        "solution": (
            "Primes less than 20: 2, 3, 5, 7, 11, 13, 17, 19.\n"
            "Count = 8.\n"
            "\\boxed{8}"
        ),
    },
    # Add hundreds/thousands of examples here for real training.
    # Good sources: MATH dataset, NuminaMath, OpenMathInstruct.
]

SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Show your reasoning step by step and put the final integer answer inside \\boxed{}."
)

def format_example(row: dict) -> dict:
    """
    Apply the GPT-OSS chat template via the tokenizer.
    Using tokenizer.apply_chat_template() is model-agnostic — it reads the
    correct special tokens (BOS, EOS, role headers) directly from the tokenizer
    config, so this code works unchanged if you swap the base model.

    SFTTrainer expects a single string field ("text") containing the full
    formatted conversation including special tokens.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": row["problem"]},
        {"role": "assistant", "content": row["solution"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,         # return a string, not token IDs
        add_generation_prompt=False,
        # ^ False = include the assistant turn (we want to train on it).
        #   True  = stop after the assistant header (used at inference time).
    )
    return {"text": text}

raw_dataset = Dataset.from_list(RAW_DATA)
dataset = raw_dataset.map(format_example, remove_columns=["problem", "solution"])
# dataset now has one column: "text" containing the full formatted conversation.


# ──────────────────────────────────────────────
# SECTION 4: Training with SFTTrainer
# ──────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,

    args=SFTConfig(
        # ── Output ──────────────────────────────
        output_dir="./sft_output",
        # ^ Directory where checkpoints and logs are saved.

        # ── Core hyperparameters ─────────────────
        num_train_epochs=3,
        # ^ Number of full passes over the training dataset.
        #   1–3 epochs is typical for SFT; more can cause overfitting.

        per_device_train_batch_size=2,
        # ^ Number of examples per GPU per forward pass.
        #   Keep low (1–4) when fine-tuning large models to avoid OOM.

        gradient_accumulation_steps=8,
        # ^ Accumulate gradients over N steps before updating weights.
        #   Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
        #   Here: 2 × 8 = 16 effective batch size.
        #   Use this to simulate larger batches without extra VRAM.

        # ── Learning rate schedule ───────────────
        learning_rate=2e-4,
        # ^ Peak learning rate for the LoRA adapters.
        #   LoRA typically uses a higher LR than full fine-tuning (1e-5 to 3e-4).
        #   2e-4 is a common starting point for r=16.

        lr_scheduler_type="cosine",
        # ^ Learning rate schedule shape.
        #   "cosine"  = decays from peak to ~0 following a cosine curve (recommended).
        #   "linear"  = linear decay.
        #   "constant"= fixed LR (no decay).

        warmup_ratio=0.05,
        # ^ Fraction of total steps used for linear LR warm-up.
        #   0.05 = first 5% of steps ramp LR from 0 → peak.
        #   Prevents unstable updates at the start of training.

        # ── Precision & memory ───────────────────
        fp16=not torch.cuda.is_bf16_supported(),
        # ^ Use float16 mixed precision if bfloat16 is not supported.

        bf16=torch.cuda.is_bf16_supported(),
        # ^ Use bfloat16 mixed precision if GPU supports it (Ampere+).
        #   bf16 is more numerically stable than fp16 for training.

        optim="adamw_8bit",
        # ^ Optimizer.
        #   "adamw_8bit"  = 8-bit AdamW (bitsandbytes). Saves ~75% optimizer VRAM.
        #   "adamw_torch" = standard PyTorch AdamW (full precision, more VRAM).
        #   "paged_adamw_8bit" = same but with paging for CPU offload.

        weight_decay=0.01,
        # ^ L2 regularization on model weights. Helps prevent overfitting.
        #   Typical range: 0.0 to 0.1.

        max_grad_norm=1.0,
        # ^ Gradient clipping: if gradient norm > this value, scale it down.
        #   Prevents exploding gradients. 1.0 is a safe default.

        # ── Sequence packing ─────────────────────
        dataset_text_field="text",
        # ^ Name of the column in `dataset` that contains the formatted text.

        max_seq_length=MAX_SEQ_LEN,
        # ^ Truncate any example longer than this. Must match model config above.

        packing=True,
        # ^ True  = pack multiple short examples into one sequence up to max_seq_length.
        #           Maximizes GPU utilization. Recommended for short datasets.
        # ^ False = pad each example to max_seq_length separately (wastes compute).

        # ── Logging & saving ─────────────────────
        logging_steps=10,
        # ^ Log training loss to console / W&B every N steps.

        save_strategy="epoch",
        # ^ When to save checkpoints.
        #   "epoch"  = save at end of each epoch.
        #   "steps"  = save every `save_steps` steps.
        #   "no"     = never save during training (only save at the end).

        save_total_limit=2,
        # ^ Keep only the N most recent checkpoints. Older ones are deleted.

        seed=42,

        report_to="none",
        # ^ Where to send training metrics.
        #   "none"   = no external logging.
        #   "wandb"  = Weights & Biases (requires: pip install wandb; wandb login).
        #   "tensorboard" = TensorBoard.
    ),
)

# Train
trainer_stats = trainer.train()
print(f"Training complete. Steps: {trainer_stats.global_step}, Loss: {trainer_stats.training_loss:.4f}")


# ──────────────────────────────────────────────
# SECTION 5: Save the LoRA Adapter
# ──────────────────────────────────────────────

# Option A: Save adapter weights only (small, ~50–200 MB for r=16).
# Load later with: PeftModel.from_pretrained(base_model, "./lora_adapter")
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# Option B: Merge LoRA into base weights and save as a full model.
# Produces a standard HuggingFace model directory (large, same size as base).
# Use this when you want to serve the model with vLLM.
model_merged = model.merge_and_unload()
# ^ Folds the LoRA matrices (A × B) back into the original weights.
#   Result: a standalone model with no PEFT dependency.

model_merged.save_pretrained("./merged_model", safe_serialization=True)
tokenizer.save_pretrained("./merged_model")
# Now serve with vLLM: vllm serve ./merged_model

# Option C: Push adapter or merged model to HuggingFace Hub.
# model.push_to_hub("your-hf-username/my-math-lora", token="hf_...")
# tokenizer.push_to_hub("your-hf-username/my-math-lora", token="hf_...")


# ──────────────────────────────────────────────
# SECTION 6: Quick Inference Test (after training)
# ──────────────────────────────────────────────
FastLanguageModel.for_inference(model)
# ^ Switches the model from training mode to optimized inference mode.
#   Must call this before generating — unsloth applies faster attention kernels.

test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "What is 15 squared minus 100?"},
]
test_prompt = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,
    # ^ True = append the assistant role header so the model knows to generate
    #   a response. GPT-OSS tokenizer inserts the correct header automatically.
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    # ^ Generate up to 256 new tokens (not counting the prompt).

    temperature=0.7,
    # ^ Sampling temperature. 0 = greedy (deterministic).

    do_sample=True,
    # ^ True  = sample from the distribution (use with temperature/top_p).
    # ^ False = greedy decoding (ignores temperature).

    top_p=0.9,
    # ^ Nucleus sampling: only consider tokens summing to top_p probability.

    use_cache=True,
    # ^ Cache key/value states for faster autoregressive generation.
    #   Always True for inference.
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# ^ Decode only the newly generated tokens (slice off the prompt).
print("Model response:", response)
