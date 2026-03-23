#!/usr/bin/env bash
# =============================================================================
# Nebius AI GPU VM — One-Shot LLM SFT Environment Setup
# =============================================================================
# Run this ONCE right after SSH-ing into a fresh Nebius instance:
#   bash setup.sh
#
# Tested on: Ubuntu 22.04, CUDA 12.8, H200 150GB
# Verified working versions (2026-03-22):
#   torch 2.10.0+cu128  |  transformers 4.57.6  |  trl 0.24.0
#   peft 0.18.1         |  accelerate 1.6.0     |  bitsandbytes 0.49.2
#   datasets 4.3.0      |  unsloth 2026.3.10    |  vllm 0.18.0
#   wandb 0.25.1        |  tensorboard 2.20.0   |  jupyter 5.9.1
# Time: ~10–15 minutes on a fast GPU node
# =============================================================================

set -euo pipefail   # exit on any error, undefined var, or pipe failure

# ── 0. Detect CUDA version ───────────────────────────────────────────────────
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "12.8")
CUDA_SHORT="${CUDA_VER//.}"   # e.g. "12.8" → "128"
echo "Detected CUDA: ${CUDA_VER}  (short: ${CUDA_SHORT})"


# ── 1. System packages ───────────────────────────────────────────────────────
echo ">>> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git git-lfs \
    tmux htop nvtop \
    build-essential \
    python3-pip python3-venv \
    wget curl unzip \
    libssl-dev libffi-dev
    # git-lfs  : needed for HuggingFace model downloads (large binary files)
    # nvtop    : GPU usage monitor (like htop but for GPUs)
    # tmux     : keep training running after SSH disconnect (see tmux_guide.sh)

git lfs install --skip-repo


# ── 2. Python virtual environment ───────────────────────────────────────────
ENV_DIR="$HOME/llm_env"
echo ">>> Creating venv at ${ENV_DIR}..."

python3 -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools -q


# ── 3. PyTorch (CUDA-matched build) ─────────────────────────────────────────
echo ">>> Installing PyTorch for CUDA ${CUDA_VER}..."

# Map common CUDA versions to PyTorch index URLs.
# PyTorch ships separate wheels per CUDA version.
if [[ "${CUDA_SHORT}" == "128" || "${CUDA_SHORT}" == "126" || "${CUDA_SHORT}" == "127" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif [[ "${CUDA_SHORT}" == "124" || "${CUDA_SHORT}" == "125" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "${CUDA_SHORT}" == "121" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "${CUDA_SHORT}" == "118" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"   # fallback to latest
fi

pip install torch torchvision torchaudio \
    --index-url "${TORCH_INDEX}" -q

echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count      : $(python -c 'import torch; print(torch.cuda.device_count())')"


# ── 4. Unsloth (QLoRA-optimized fine-tuning) ────────────────────────────────
echo ">>> Installing unsloth..."
# Unsloth provides fast LoRA loading + patched attention kernels.
# Must be installed AFTER PyTorch so it picks up the correct CUDA kernels.
# Install mergekit first — required by trl CLI (DPOTrainer dependency)
pip install mergekit -q
pip install "unsloth[cu${CUDA_SHORT}-torch260] @ git+https://github.com/unslothai/unsloth.git" -q \
    || pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" -q
# ^ Fallback to generic install if versioned wheel isn't available yet.


# ── 5. Core training stack ───────────────────────────────────────────────────
echo ">>> Installing training dependencies..."
pip install -q \
    "transformers>=4.47.0" \
    "trl>=0.12.0" \
    "peft>=0.14.0" \
    "accelerate>=1.2.0" \
    "bitsandbytes>=0.45.0" \
    "datasets>=3.0.0" \
    sentencepiece \
    protobuf \
    scipy numpy
# transformers : HuggingFace model hub + tokenizers     (verified: 4.57.6)
# trl          : SFTTrainer, DPOTrainer, GRPOTrainer    (verified: 0.24.0)
# peft         : LoRA / QLoRA adapter management         (verified: 0.18.1)
# accelerate   : Multi-GPU / mixed-precision training    (verified: 1.6.0)
# bitsandbytes : 4-bit / 8-bit quantization              (verified: 0.49.2)
# datasets     : HuggingFace dataset loading             (verified: 4.3.0)
# sentencepiece: Tokenizer backend (LLaMA, Mistral, etc.)
# protobuf     : Required by some tokenizer configs


# ── 6. vLLM (fast inference server) ─────────────────────────────────────────
echo ">>> Installing vLLM..."
# vLLM is used for fast inference AFTER training (same as the AIMO notebook).
# Install separately — it has strict torch/cuda version requirements.
# Verified: vllm 0.18.0 works with torch 2.10.0+cu128
pip install vllm -q


# ── 7. Experiment tracking ───────────────────────────────────────────────────
echo ">>> Installing experiment tracking tools..."
pip install -q \
    wandb \
    tensorboard \
    jupyter jupyterlab ipywidgets
# wandb       : Weights & Biases — log loss, GPU stats, hyperparams (verified: 0.25.1)
#               Usage: wandb login   (paste API key from wandb.ai/settings)
# tensorboard : Local alternative to W&B (verified: 2.20.0)
#               View with: tensorboard --logdir ./runs
# jupyter     : Optional — run notebooks on GPU server (verified: 5.9.1)
#               Access via SSH tunnel: ssh -L 8888:localhost:8888 user@<ip>


# ── 8. HuggingFace CLI (model downloads) ────────────────────────────────────
echo ">>> Installing huggingface_hub CLI..."
pip install -q huggingface_hub
# Usage after setup:
#   huggingface-cli login          # authenticate once
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
#   huggingface-cli download openai/gpt-oss-20b --local-dir ~/model/gpt-oss-20b


# ── 9. Clone your repo ───────────────────────────────────────────────────────
REPO_DIR="$HOME/Journey_of_MachineLearning"
if [ ! -d "${REPO_DIR}" ]; then
    echo ">>> Cloning your ML repo..."
    git clone https://github.com/yihuang1995/Journey_of_MachineLearning.git "${REPO_DIR}"
else
    echo ">>> Repo already exists, pulling latest..."
    git -C "${REPO_DIR}" pull
fi


# ── 10. Shell convenience: auto-activate venv on login ───────────────────────
BASHRC="$HOME/.bashrc"
if ! grep -q "llm_env" "${BASHRC}"; then
    echo "" >> "${BASHRC}"
    echo "# Auto-activate LLM environment" >> "${BASHRC}"
    echo "source ${ENV_DIR}/bin/activate" >> "${BASHRC}"
    echo "export TOKENIZERS_PARALLELISM=false" >> "${BASHRC}"
    echo "export CUDA_VISIBLE_DEVICES=0" >> "${BASHRC}"
    # ^ Set to "0,1" for 2 GPUs, "0,1,2,3" for 4 GPUs, etc.
fi


# ── 11. Verify GPU setup ─────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Environment check"
echo "============================================"
python - <<'EOF'
import unsloth  # must be first to apply patches
import torch, transformers, peft, trl, vllm
import wandb, datasets, accelerate, bitsandbytes
print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"trl          : {trl.__version__}")
print(f"vllm         : {vllm.__version__}")
print(f"unsloth      : {unsloth.__version__}")
print(f"wandb        : {wandb.__version__}")
print(f"datasets     : {datasets.__version__}")
print(f"accelerate   : {accelerate.__version__}")
print(f"bitsandbytes : {bitsandbytes.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPU count    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}      : {torch.cuda.get_device_name(i)}  ({mem:.0f} GB)")
EOF


# ── 12. Optional: Start Jupyter on a background port ─────────────────────────
# Uncomment to auto-launch Jupyter on port 8888.
# Access via SSH tunnel: ssh -L 8888:localhost:8888 user@<nebius-ip>
#
# nohup jupyter lab \
#     --ip=0.0.0.0 \
#     --port=8888 \
#     --no-browser \
#     --NotebookApp.token='nebius' \
#     > "$HOME/jupyter.log" 2>&1 &
# echo "Jupyter running — tunnel with: ssh -L 8888:localhost:8888 user@<ip>"


echo ""
echo "============================================"
echo " Setup complete! Next steps:"
echo "  1. source ~/.bashrc          (activate env in current shell)"
echo "  2. wandb login               (optional: experiment tracking)"
echo "  3. huggingface-cli login     (for gated models like Llama)"
echo "  4. cd ~/Journey_of_MachineLearning/\"Chapter LLM SFT\""
echo "  5. python sft_example.py     (start training!)"
echo "============================================"
echo ""
echo "Notes:"
echo "  - gpt-oss-20b is MxFP4 quantized — do NOT use load_in_4bit=True"
echo "    Load with: AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)"
echo "  - Use tmux to keep training alive: tmux new -s train"
echo "  - Monitor GPU: watch -n 2 nvidia-smi  OR  nvtop"
