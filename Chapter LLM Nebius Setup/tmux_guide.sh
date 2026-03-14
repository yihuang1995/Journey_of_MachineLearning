#!/usr/bin/env bash
# =============================================================================
# tmux Survival Guide for Long-Running GPU Training Jobs
# =============================================================================
# Problem: SSH connections drop → training process dies.
# Solution: Run training inside a tmux session — it keeps running on the server
#           even after you disconnect. Reconnect anytime to check progress.
# =============================================================================

# ── QUICK REFERENCE ──────────────────────────────────────────────────────────
#
#  START a new session:
#    tmux new -s train
#              └── "train" is the session name (choose anything)
#
#  DETACH (leave session running, go back to normal shell):
#    Ctrl+b  then  d
#
#  LIST running sessions:
#    tmux ls
#
#  REATTACH to a running session:
#    tmux attach -t train
#
#  KILL a session:
#    tmux kill-session -t train
#
# ── WITHIN A SESSION ─────────────────────────────────────────────────────────
#
#  Split screen horizontally (top/bottom):
#    Ctrl+b  then  "
#
#  Split screen vertically (left/right):
#    Ctrl+b  then  %
#
#  Switch between panes:
#    Ctrl+b  then  arrow key
#
#  Scroll up to read logs:
#    Ctrl+b  then  [    (enter scroll mode)
#    q                  (exit scroll mode)
#
# ── TYPICAL TRAINING WORKFLOW ────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  Typical Training Workflow on Nebius"
echo "========================================================"
echo ""
echo "Step 1: SSH into Nebius"
echo "  ssh -i ~/.ssh/id_nebius <user>@<nebius-ip>"
echo ""
echo "Step 2: Start a tmux session"
echo "  tmux new -s train"
echo ""
echo "Step 3: Activate environment & launch training"
echo "  source ~/llm_env/bin/activate"
echo "  cd ~/Journey_of_MachineLearning/\"Chapter LLM SFT\""
echo "  python sft_example.py 2>&1 | tee training.log"
echo "  #                           └─ save stdout+stderr to file"
echo ""
echo "Step 4: Detach and close your laptop"
echo "  Ctrl+b, then d"
echo ""
echo "Step 5: Check GPU usage from another pane/terminal"
echo "  watch -n 2 nvidia-smi"
echo "  # or use nvtop for a nicer UI:"
echo "  nvtop"
echo ""
echo "Step 6: Later — reconnect and check progress"
echo "  ssh -i ~/.ssh/id_nebius <user>@<nebius-ip>"
echo "  tmux attach -t train"
echo "  # or just tail the log:"
echo "  tail -f ~/Journey_of_MachineLearning/\"Chapter LLM SFT\"/training.log"
echo ""
echo "========================================================"

# ── MULTI-GPU: launch with torchrun ──────────────────────────────────────────
echo ""
echo "Multi-GPU training (e.g. 2× H100):"
echo "  torchrun --nproc_per_node=2 sft_example.py"
echo "  # nproc_per_node = number of GPUs on this machine"
echo ""
echo "Check GPU assignments:"
echo "  echo \$CUDA_VISIBLE_DEVICES   # e.g. '0,1'"
echo "  nvidia-smi                    # see which processes use which GPU"
