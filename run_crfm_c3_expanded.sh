#!/bin/bash
# Run C3 few-shot 41-term expansion for CRFM GPT-2 Small (seed 1) on CPU.
# Two checkpoints: ck-1000 (near-random) and ck-400000 (trained).
# Expected runtime: ~2.5–4 hours total on CPU.

set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/crfm_c3_expanded_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== CRFM C3 Expanded: $(date) ===" | tee "$LOG"
echo "Device: CPU (CUDA not used)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- crfm1 / checkpoint-1000 ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model crfm1 --checkpoint checkpoint-1000 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- crfm1 / checkpoint-400000 ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model crfm1 --checkpoint checkpoint-400000 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== CRFM C3 Expanded done: $(date) ===" | tee -a "$LOG"
