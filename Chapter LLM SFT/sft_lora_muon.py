"""
Fine-Tuning with LoRA + MUON Optimizer
=======================================
MUON (MomentUm Orthogonalized by Newton-schulz) is a 2024 optimizer from
Keller Jordan et al. (https://github.com/KellerJordan/modded-nanogpt).

Key idea vs Adam:
  - Adam tracks per-parameter first + second moment estimates (needs 2× param VRAM).
  - MUON uses Nesterov momentum and then ORTHOGONALIZES the update matrix via
    Newton-Schulz iterations, which effectively "whitens" the gradient directions.
  - Result: faster loss reduction per step on matrix weights, with LESS optimizer
    state memory (only 1 momentum buffer, no second moment).

Why pair MUON with LoRA?
  - LoRA adapters are small 2D matrices (A: d×r, B: r×d) — exactly the shape
    MUON is designed for.
  - MUON is applied to the LoRA B matrices (hidden-to-hidden updates).
  - AdamW is kept for embeddings, layer-norm scales, biases (1D / non-matrix params)
    because MUON only makes sense for rank-≥2 tensors.

Install:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl peft datasets bitsandbytes
  pip install muon   # OR use the implementation in Section 2 below
"""

import math
import torch
import torch.nn as nn
from typing import Callable, Iterable


# ──────────────────────────────────────────────
# SECTION 1: Load Model + LoRA (same as sft_example.py)
# ──────────────────────────────────────────────
from unsloth import FastLanguageModel

MAX_SEQ_LEN = 2048
LORA_RANK   = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,         # auto: bfloat16 on Ampere+, float16 otherwise
    load_in_4bit=True,  # QLoRA: 4-bit base weights, LoRA adapters in bf16
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)


# ──────────────────────────────────────────────
# SECTION 2: MUON Optimizer (from scratch, annotated)
# ──────────────────────────────────────────────

def _newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration — computes the orthogonal factor of G.

    Given a matrix G, returns a matrix X such that X ≈ U (from G = U Σ Vᵀ SVD),
    i.e. X is the "direction" of G with all singular values set to 1.

    This is a matrix polynomial iteration:
        X_{k+1} = a·X_k + (b·A + c·A²) @ X_k    where A = X_k @ X_kᵀ
    Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) converge in ~5 steps.

    Why this instead of SVD?
      - SVD is O(mn·min(m,n)) and not easily batched on GPU.
      - Newton-Schulz is a fixed number of matmuls — fast on GPU/TPU.
    """
    assert G.ndim == 2, "MUON Newton-Schulz only defined for 2D matrices."

    a, b, c = 3.4445, -4.7750, 2.0315

    # Work in bfloat16 for speed; normalize so iteration is numerically stable.
    X = G.to(torch.bfloat16) / (G.norm() + eps)

    # If G is tall (more rows than cols), transpose so X is wide — iteration
    # converges faster when X has more columns than rows.
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T                    # A = X Xᵀ  (square, cheap)
        X = a * X + (b * A + c * A @ A) @ X   # polynomial update

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    MUON — MomentUm Orthogonalized by Newton-schulz.

    Algorithm per step for each 2D parameter W:
      1. Accumulate Nesterov momentum:
            buf = momentum * buf + grad               (momentum buffer update)
            update = grad + momentum * buf            (Nesterov lookahead)
      2. Orthogonalize:
            update = newtonschulz5(update)            (set all singular values → 1)
      3. Scale and apply:
            W = W - lr * update * scale_factor

    For non-2D parameters (biases, scalars), falls back to plain SGD.

    Args:
        params      : Parameters to optimize (should be 2D weight matrices).
        lr          : Learning rate. MUON typically needs a HIGHER lr than Adam
                      because the orthogonalized update has unit-scale singular values.
                      Typical range: 0.01 – 0.1.
        momentum    : Nesterov momentum coefficient. Range: 0.9 – 0.98.
        nesterov    : Use Nesterov momentum (True = standard MUON, False = plain SGD).
        ns_steps    : Number of Newton-Schulz iterations. 5 is sufficient in practice.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # Initialize momentum buffer on first step.
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]

                # Nesterov momentum update:
                #   buf  ← momentum * buf + grad
                #   g    ← grad + momentum * buf    (lookahead gradient)
                buf.mul_(momentum).add_(grad)
                g = grad.add(buf, alpha=momentum) if nesterov else buf

                if g.ndim == 2:
                    # ── MUON path: orthogonalize the 2D update ──────────────
                    g = _newtonschulz5(g, steps=ns_steps)

                    # Scale the orthogonalized update so its RMS matches the
                    # original gradient (prevents step-size blow-up).
                    scale = max(1, g.size(0) / g.size(1)) ** 0.5
                    # ^ For a tall matrix (more rows than cols), singular value
                    #   orthogonalization amplifies the update — compensate here.

                    p.add_(g, alpha=-lr * scale)

                else:
                    # ── Fallback: plain SGD for 1D params (biases, norms) ───
                    p.add_(g, alpha=-lr)

        return loss


# ──────────────────────────────────────────────
# SECTION 3: Split Parameters — MUON vs AdamW
# ──────────────────────────────────────────────
# MUON is only meaningful for 2D matrices.
# Adam handles everything else (embeddings are 2D but too large to orthogonalize
# cheaply, so we keep them under Adam too).

muon_params  = []   # LoRA B matrices — 2D, small, good MUON targets
adam_params  = []   # everything else: A matrices, embeddings, norms, biases

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue   # frozen base-model weights — skip

    if (
        "lora_B" in name          # LoRA output matrix (r → d_out): best MUON target
        and param.ndim == 2
        and param.size(0) >= 2    # must be a proper matrix, not a degenerate vector
        and param.size(1) >= 2
    ):
        muon_params.append(param)
        print(f"[MUON ] {name:60s} shape={tuple(param.shape)}")

    else:
        adam_params.append(param)
        print(f"[Adam ] {name:60s} shape={tuple(param.shape)}")

print(f"\nMUON params : {len(muon_params)}")
print(f"Adam  params : {len(adam_params)}\n")


# ──────────────────────────────────────────────
# SECTION 4: Build Dual Optimizer
# ──────────────────────────────────────────────

muon_optimizer = Muon(
    muon_params,
    lr=0.02,
    # ^ MUON uses a much higher LR than Adam because the orthogonalized update
    #   has singular values ~1 (no second-moment dampening).
    #   Typical range for LoRA B matrices: 0.01 – 0.05.

    momentum=0.95,
    # ^ Nesterov momentum. Higher = smoother updates but slower to react to
    #   loss landscape changes. 0.95 is the MUON paper default.

    nesterov=True,
    # ^ Use Nesterov look-ahead (standard MUON). Set False for vanilla momentum.

    ns_steps=5,
    # ^ Newton-Schulz iterations. 5 converges well for matrices up to ~4096×4096.
    #   Increase to 7–10 for very large matrices or if you see instability.
)

adam_optimizer = torch.optim.AdamW(
    adam_params,
    lr=2e-4,
    # ^ Standard LoRA Adam LR. Lower than MUON because Adam uses second-moment
    #   dampening which keeps effective step sizes controlled.

    betas=(0.9, 0.95),
    # ^ (β1, β2): exponential decay rates for 1st and 2nd moment estimates.
    #   β1=0.9  → momentum for mean gradient (smooths noise).
    #   β2=0.95 → momentum for variance estimate (0.95 instead of the default
    #             0.999 — faster adaptation to changing curvature, common for LLMs).

    eps=1e-8,
    # ^ Small constant added to denominator for numerical stability.

    weight_decay=0.1,
    # ^ L2 regularization. Higher than the MUON path because Adam params include
    #   embeddings and layer-norm scales where regularization helps more.
)


# ──────────────────────────────────────────────
# SECTION 5: LR Scheduler (cosine with warmup)
# ──────────────────────────────────────────────
from torch.optim.lr_scheduler import LambdaLR

TOTAL_STEPS   = 500
WARMUP_STEPS  = 25   # 5% of total

def cosine_with_warmup(step: int) -> float:
    """Returns a multiplier in [0, 1] for the current step."""
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)       # linear warm-up
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

muon_scheduler = LambdaLR(muon_optimizer, lr_lambda=cosine_with_warmup)
adam_scheduler = LambdaLR(adam_optimizer, lr_lambda=cosine_with_warmup)


# ──────────────────────────────────────────────
# SECTION 6: Dataset + Collator
# ──────────────────────────────────────────────
from datasets import Dataset
from torch.utils.data import DataLoader

SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Show your reasoning step by step and put the final integer answer inside \\boxed{}."
)

RAW_DATA = [
    {"problem": "What is the sum of all integers from 1 to 100?",
     "solution": "Using n(n+1)/2: 100×101/2 = 5050.\n\\boxed{5050}"},
    {"problem": "How many primes are less than 20?",
     "solution": "Primes: 2,3,5,7,11,13,17,19 → 8 primes.\n\\boxed{8}"},
    # Add more examples here.
]

def format_example(row):
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{row['problem']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{row['solution']}<|eot_id|>"
    )
    return {"text": text}

dataset = Dataset.from_list(RAW_DATA).map(format_example, remove_columns=["problem", "solution"])

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    enc["labels"] = enc["input_ids"].clone()
    # ^ For causal LM training, labels = input_ids (next-token prediction).
    #   The loss is computed on all tokens; mask the prompt if you want to
    #   train only on the completion (set prompt token labels to -100).
    return enc

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)


# ──────────────────────────────────────────────
# SECTION 7: Custom Training Loop
# ──────────────────────────────────────────────
# We use a manual loop instead of SFTTrainer because Trainer only accepts
# a single optimizer — here we have two (MUON + Adam).

GRAD_ACCUM_STEPS = 4
# ^ Accumulate gradients over N micro-batches before stepping.
#   Effective batch size = batch_size × GRAD_ACCUM_STEPS = 2 × 4 = 8.

MAX_GRAD_NORM = 1.0
# ^ Gradient clipping threshold. Prevents exploding gradients.
#   Applied BEFORE calling optimizer.step().

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

global_step = 0
accum_loss  = 0.0

for epoch in range(3):
    for micro_step, batch in enumerate(dataloader):

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        # ^ Divide by accum steps so the gradient magnitude is consistent
        #   regardless of how many micro-batches we accumulate.

        loss.backward()
        accum_loss += loss.item()

        # Step only after accumulating GRAD_ACCUM_STEPS micro-batches.
        if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:

            # Gradient clipping (applied to ALL trainable params together).
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                MAX_GRAD_NORM,
            )

            # Step both optimizers.
            muon_optimizer.step()
            adam_optimizer.step()

            # Step both schedulers.
            muon_scheduler.step()
            adam_scheduler.step()

            # Zero gradients for both param groups.
            muon_optimizer.zero_grad()
            adam_optimizer.zero_grad()

            global_step += 1

            if global_step % 10 == 0:
                muon_lr = muon_scheduler.get_last_lr()[0]
                adam_lr  = adam_scheduler.get_last_lr()[0]
                print(
                    f"Step {global_step:4d} | loss {accum_loss:.4f} "
                    f"| muon_lr {muon_lr:.2e} | adam_lr {adam_lr:.2e}"
                )

            accum_loss = 0.0

            if global_step >= TOTAL_STEPS:
                break

    if global_step >= TOTAL_STEPS:
        break

print("Training complete.")


# ──────────────────────────────────────────────
# SECTION 8: Save (identical to sft_example.py)
# ──────────────────────────────────────────────
# Option A: adapter only
model.save_pretrained("./lora_muon_adapter")
tokenizer.save_pretrained("./lora_muon_adapter")

# Option B: merge + save full model (ready for vLLM)
merged = model.merge_and_unload()
merged.save_pretrained("./merged_muon_model", safe_serialization=True)
tokenizer.save_pretrained("./merged_muon_model")


# ──────────────────────────────────────────────
# SECTION 9: When to Use MUON vs AdamW
# ──────────────────────────────────────────────
"""
                  MUON                          AdamW
─────────────────────────────────────────────────────────────────
Best for      2D weight matrices (Linear)    All parameter shapes
Optimizer     1 momentum buffer only         2 moment buffers (2× VRAM)
Memory        ~½ Adam memory per param       baseline
Update        orthogonal (unit singular      scaled by 1/√(v̂) curvature
              values, equal directions)      estimate
LR scale      needs much higher LR (~0.02)   typically 1e-4 to 3e-4
Convergence   faster loss drop per step      stable, well-understood
Caveats       only 2D; no per-param adapt.   can be slow for rank-def.
              may need LR tuning             gradients

Recommendation:
  - Use MUON for LoRA B matrices (hidden→hidden projections).
  - Use AdamW for LoRA A matrices, embeddings, norms, biases.
  - If you want simplicity, just use AdamW everywhere (sft_example.py).
  - If you want speed/efficiency on matrix params, add MUON (this file).
"""
