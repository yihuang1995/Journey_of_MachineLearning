#!/usr/bin/env bash
# =============================================================================
# Nebius AI GPU VM — One-Shot LLM SFT Environment Setup
# =============================================================================
# Run this ONCE right after SSH-ing into a fresh Nebius instance:
#   bash setup.sh
#
# Tested on: Ubuntu 22.04, CUDA 12.4, H100 / A100 80GB
# Time: ~10–15 minutes on a fast GPU node
# =============================================================================

set -euo pipefail   # exit on any error, undefined var, or pipe failure

# ── 0. Detect CUDA version ───────────────────────────────────────────────────
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "12.4")
CUDA_SHORT="${CUDA_VER//.}"   # e.g. "12.4" → "124"
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
if [[ "${CUDA_SHORT}" == "124" || "${CUDA_SHORT}" == "125" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "${CUDA_SHORT}" == "121" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "${CUDA_SHORT}" == "118" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"   # fallback
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
pip install "unsloth[cu${CUDA_SHORT}-torch260] @ git+https://github.com/unslothai/unsloth.git" -q \
    || pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" -q
# ^ Fallback to generic install if versioned wheel isn't available yet.


# ── 5. Core training stack ───────────────────────────────────────────────────
echo ">>> Installing training dependencies..."
pip install -q \
    transformers>=4.47.0 \
    # ^ HuggingFace model hub + tokenizers
    trl>=0.12.0 \
    # ^ SFTTrainer, DPOTrainer, GRPOTrainer (RLHF tools)
    peft>=0.14.0 \
    # ^ LoRA / QLoRA adapter management
    accelerate>=1.2.0 \
    # ^ Multi-GPU / mixed-precision training wrapper
    bitsandbytes>=0.45.0 \
    # ^ 4-bit / 8-bit quantization + 8-bit AdamW optimizer
    datasets>=3.0.0 \
    # ^ HuggingFace dataset loading and processing
    sentencepiece \
    # ^ Tokenizer backend for many models (LLaMA, Mistral, etc.)
    protobuf \
    # ^ Required by some tokenizer configs
    scipy numpy


# ── 6. vLLM (fast inference server) ─────────────────────────────────────────
echo ">>> Installing vLLM..."
# vLLM is used for fast inference AFTER training (same as the AIMO notebook).
# Install separately — it has strict torch/cuda version requirements.
pip install vllm -q


# ── 7. Experiment tracking ───────────────────────────────────────────────────
echo ">>> Installing experiment tracking tools..."
pip install -q \
    wandb \
    # ^ Weights & Biases: log loss curves, GPU stats, hyperparams. Free tier available.
    #   Usage: wandb login   (paste your API key from wandb.ai/settings)
    tensorboard \
    # ^ Local alternative to W&B. View with: tensorboard --logdir ./runs
    jupyter jupyterlab ipywidgets
    # ^ Optional: run notebooks on the GPU server (see JUPYTER section below)


# ── 8. HuggingFace CLI (model downloads) ────────────────────────────────────
echo ">>> Installing huggingface_hub CLI..."
pip install -q huggingface_hub
# Usage after setup:
#   huggingface-cli login          # authenticate once
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct


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
import torch, transformers, peft, trl, vllm
print(f"torch       : {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"peft        : {peft.__version__}")
print(f"trl         : {trl.__version__}")
print(f"vllm        : {vllm.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
print(f"GPU count   : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}     : {torch.cuda.get_device_name(i)}  ({mem:.0f} GB)")
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
echo "  3. huggingface-cli login     (for gated models)"
echo "  4. cd ~/Journey_of_MachineLearning/\"Chapter LLM SFT\""
echo "  5. python sft_example.py     (start training!)"
echo "============================================"
