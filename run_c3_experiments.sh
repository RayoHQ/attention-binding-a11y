#!/usr/bin/env bash
# Sequential runner for all C3 GPU experiments.
# Runs paraphrase control (3 runs) then C3 expansion (6 Pythia runs).
# Results written to:
#   data/results/few_shot_c3_paraphrase/
#   data/results/few_shot_c3_expanded/
set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/c3_experiments.log
mkdir -p logs

echo "=== C3 experiments start: $(date) ===" | tee -a "$LOG"

echo "--- Paraphrase control ---" | tee -a "$LOG"
python src/eval_c3_paraphrase.py --model 160m  --checkpoint step15000  2>&1 | tee -a "$LOG"
python src/eval_c3_paraphrase.py --model 1b    --checkpoint step15000  2>&1 | tee -a "$LOG"
python src/eval_c3_paraphrase.py --model 2.8b  --checkpoint step15000  2>&1 | tee -a "$LOG"

echo "--- C3 expanded (41 terms) ---" | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model 160m  --checkpoint step15000   2>&1 | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model 160m  --checkpoint step143000  2>&1 | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model 1b    --checkpoint step15000   2>&1 | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model 1b    --checkpoint step143000  2>&1 | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model 2.8b  --checkpoint step15000   2>&1 | tee -a "$LOG"
python src/eval_few_shot_c3_expanded.py --model 2.8b  --checkpoint step143000  2>&1 | tee -a "$LOG"

echo "=== C3 experiments done: $(date) ===" | tee -a "$LOG"
