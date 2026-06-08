#!/bin/bash
# Full GPU pipeline: wave-2 behavioral → C5 canonical → analysis re-run
# Run after extract_binding_wave2.py completes.
set -e
cd /teamspace/studios/this_studio/attention-binding-a11y

echo "=== [1/3] Wave-2 behavioral evaluation ===" && \
python src/eval_behavior_wave2.py --all 2>&1 | tee logs/wave2_behavioral.log

echo "=== [2/3] C5 canonical 41-term ablation ===" && \
python src/run_c5_canonical.py --all 2>&1 | tee logs/c5_canonical.log

echo "=== [3/3] Re-run C1-B and C4-B analysis ===" && \
python src/analyze_c1b_within_term.py 2>&1 | tee logs/c1b_final.log && \
python src/analyze_c4b_decoupling.py  2>&1 | tee logs/c4b_final.log

echo "=== ALL DONE ==="
