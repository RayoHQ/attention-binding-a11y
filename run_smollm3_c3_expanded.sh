#!/bin/bash
# Run C3 few-shot 41-term expansion for SmolLM3-3B on GPU.
# Two checkpoints: step40k (early) and step3440k (late).

set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/smollm3_c3_expanded_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== SmolLM3-3B C3 Expanded: $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo "--- smollm3 / step40k ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model smollm3 --checkpoint step40k 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- smollm3 / step3440k ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model smollm3 --checkpoint step3440k 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== SmolLM3-3B C3 Expanded done: $(date) ===" | tee -a "$LOG"
