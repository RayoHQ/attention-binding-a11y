#!/bin/bash
# Queue all remaining GPU jobs after 2.8b binding finishes.
# Run from project root: bash run_remaining_jobs.sh
set -e

echo "=== Job 1: Behavioral eval — 160m (21 new terms) ==="
python src/eval_behavior_tier123.py --model 160m 2>&1 | tee logs/behavioral_tier123_160m.log

echo "=== Job 2: Behavioral eval — 1b (21 new terms) ==="
python src/eval_behavior_tier123.py --model 1b 2>&1 | tee logs/behavioral_tier123_1b.log

echo "=== Job 3: Behavioral eval — 2.8b (21 new terms) ==="
python src/eval_behavior_tier123.py --model 2.8b 2>&1 | tee logs/behavioral_tier123_2.8b.log

echo "=== Job 4: OLMo-1B binding extraction (9 original terms, 8 checkpoints) ==="
python src/extract_binding_olmo.py --all 2>&1 | tee logs/binding_olmo_1b.log

echo "=== All jobs complete ==="
ls -lh data/results/binding_tier123/ | wc -l
ls -lh data/results/behavioral_tier123/ | wc -l
ls -lh data/results/binding/ | grep olmo | wc -l
