#!/bin/bash
# Run C5 canonical41 ablation with per-term breakdown for all new models.
# Outputs: data/results/causal/*_c5_canonical41_perterm.json
# Models: OLMo-1B, SmolLM3-3B, Qwen2.5-1.5B, CRFM seeds 1-5

set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/c5_perterm_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== C5 Per-Term Ablation: $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo "--- OLMo-1B step143k ---" | tee -a "$LOG"
python src/run_c5_new_models.py --model olmo 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- SmolLM3-3B step3440k ---" | tee -a "$LOG"
python src/run_c5_new_models.py --model smollm3 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Qwen2.5-1.5B final ---" | tee -a "$LOG"
python src/run_c5_new_models.py --model qwen 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- CRFM all seeds (1-5) ---" | tee -a "$LOG"
python src/run_c5_new_models.py --model crfm 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== C5 Per-Term done: $(date) ===" | tee -a "$LOG"
