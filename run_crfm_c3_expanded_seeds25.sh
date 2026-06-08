#!/bin/bash
# Run C3 few-shot 41-term expansion for CRFM GPT-2 Small seeds 2-5 on CPU.
# Two checkpoints per seed: checkpoint-1000 (early) and checkpoint-400000 (late).

set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/crfm_c3_expanded_seeds25_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== CRFM C3 Expanded seeds 2-5: $(date) ===" | tee "$LOG"
echo "Device: CPU (CUDA not used)" | tee -a "$LOG"

for SEED in 2 3 4 5; do
    echo "" | tee -a "$LOG"
    echo "--- crfm${SEED} / checkpoint-1000 ---" | tee -a "$LOG"
    python src/eval_few_shot_c3_expanded.py --model crfm${SEED} --checkpoint checkpoint-1000 2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "--- crfm${SEED} / checkpoint-400000 ---" | tee -a "$LOG"
    python src/eval_few_shot_c3_expanded.py --model crfm${SEED} --checkpoint checkpoint-400000 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== CRFM C3 Expanded seeds 2-5 done: $(date) ===" | tee -a "$LOG"
