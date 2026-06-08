#!/bin/bash
# Run C3 few-shot 41-term expansion for OLMo-1B on GPU.
# Two checkpoints: step15k (early) and step143k (late).

set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/olmo_c3_expanded_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== OLMo-1B C3 Expanded: $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo "--- olmo / step15k ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model olmo --checkpoint step15k 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- olmo / step143k ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model olmo --checkpoint step143k 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== OLMo-1B C3 Expanded done: $(date) ===" | tee -a "$LOG"
