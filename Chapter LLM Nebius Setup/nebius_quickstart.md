# Nebius AI GPU VM — LLM SFT Quick-Start Guide

## 1. Create a VM on Nebius Console

1. Go to [console.nebius.ai](https://console.nebius.ai) → **Compute** → **Create Instance**
2. Recommended config for SFT:

| Field | Recommended |
|---|---|
| Image | Ubuntu 22.04 LTS with CUDA 12.x |
| GPU | H100 SXM5 80GB × 1 (small models / QLoRA) or ×2–4 (120B) |
| CPU | 16–32 vCPUs |
| RAM | 128–256 GB |
| Disk | 500 GB SSD (models are large — Llama 70B ≈ 140 GB, 120B ≈ 240 GB) |
| SSH key | Paste your **public** key (`~/.ssh/id_nebius.pub`) |

3. Note the **external IP** after the instance starts.

---

## 2. Connect via SSH

```bash
ssh -i ~/.ssh/id_nebius ubuntu@<nebius-ip>
```

Add to `~/.ssh/config` for convenience:
```
Host nebius
    HostName <nebius-ip>
    User ubuntu
    IdentityFile ~/.ssh/id_nebius
```
Then just: `ssh nebius`

---

## 3. One-Shot Environment Setup

Upload and run the setup script (takes ~10–15 min):

```bash
# From your Mac — copy the script to the server
scp "Chapter LLM Nebius Setup/setup.sh" nebius:~/setup.sh

# On the server — run it
ssh nebius
bash ~/setup.sh
```

What `setup.sh` does in order:
1. Installs system packages (`git-lfs`, `nvtop`, `tmux`, etc.)
2. Creates a Python venv at `~/llm_env` (auto-activated on login)
3. Installs PyTorch matched to the server's CUDA version
4. Installs **unsloth** (fast QLoRA kernels)
5. Installs **trl**, **peft**, **accelerate**, **bitsandbytes**, **datasets**
6. Installs **vLLM** for post-training inference
7. Installs **wandb** + **tensorboard** for experiment tracking
8. Clones your GitHub repo to `~/Journey_of_MachineLearning`

---

## 4. Download a Model from HuggingFace

```bash
# Authenticate once (get token from huggingface.co/settings/tokens)
huggingface-cli login

# Download — stored in ~/.cache/huggingface/hub/
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# For a specific local path:
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/models/llama-3.1-8b
```

---

## 5. Run Training (inside tmux)

Always use **tmux** so training survives SSH disconnects:

```bash
# Start a persistent session
tmux new -s train

# Activate env + start training
source ~/llm_env/bin/activate
cd ~/Journey_of_MachineLearning/"Chapter LLM SFT"
python sft_example.py 2>&1 | tee training.log

# Detach (training keeps running):  Ctrl+b, then d
# Reattach later:
tmux attach -t train
```

See `tmux_guide.sh` for more tmux commands.

---

## 6. Monitor GPU Usage

```bash
# Real-time GPU stats (refresh every 2s)
watch -n 2 nvidia-smi

# Better UI (like htop for GPUs)
nvtop

# Training log
tail -f training.log
```

Key numbers to watch in `nvidia-smi`:

| Field | What it means |
|---|---|
| `GPU-Util` | Should be 90–100% during training |
| `Memory-Usage` | Should be near the GPU's max (e.g. 75/80 GB) |
| `Temp` | Alert if >85°C — reduce batch size |

---

## 7. Multi-GPU Training

Edit `CUDA_VISIBLE_DEVICES` and use `torchrun`:

```bash
# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Launch with 2 processes (one per GPU)
torchrun --nproc_per_node=2 sft_example.py
```

In `sft_example.py`, `accelerate` and `unsloth` handle weight sharding automatically.
For very large models (120B+), also set `tensor_parallel_size` in vLLM.

---

## 8. Jupyter Lab (optional — access from browser)

```bash
# On the server — start Jupyter in background
nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --NotebookApp.token='nebius' \
    > ~/jupyter.log 2>&1 &

# On your Mac — open an SSH tunnel
ssh -L 8888:localhost:8888 nebius

# Open in browser:  http://localhost:8888   (token: nebius)
```

---

## 9. Save Your Work / Transfer Files

```bash
# Pull latest code on the server
git -C ~/Journey_of_MachineLearning pull

# Push trained adapter back to HuggingFace
huggingface-cli upload your-username/my-math-lora ./lora_adapter

# Copy files from server to your Mac
scp nebius:~/Journey_of_MachineLearning/"Chapter LLM SFT"/training.log .
scp -r nebius:~/lora_adapter ./lora_adapter_from_nebius
```

---

## 10. Shut Down When Done

**Always stop the instance when you're not training** — GPU VMs are billed per minute.

Nebius Console → Compute → Select instance → **Stop**

Or from the terminal:
```bash
# Graceful shutdown
sudo shutdown now
```
