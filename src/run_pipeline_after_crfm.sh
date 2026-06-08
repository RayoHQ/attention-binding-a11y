#!/usr/bin/env bash
# Chained pipeline after CRFM behavioral finishes.
# Run: bash src/run_pipeline_after_crfm.sh 2>&1 | tee logs/pipeline_post_crfm.log
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "POST-CRFM PIPELINE: $(date)"
echo "============================================================"

# ── Step 1: C1-B + C4-B for CRFM 45-term ──────────────────────────────────
echo ""
echo "[1/8] C1-B analysis — CRFM 45-term"
python src/analyze_c1b_within_term.py --model crfm45

echo ""
echo "[2/8] C4-B analysis — CRFM 45-term"
python src/analyze_c4b_decoupling.py --model crfm45

# ── Step 2: SmolLM3 45-term binding + behavioral (sequential, on CUDA) ────
echo ""
echo "[3/8] SmolLM3 45-term binding (8 ck × 451 prompts)"
python src/extract_binding_smollm3.py --all \
    --prompts data/prompts/canonical_45terms.jsonl \
    --outdir data/results/binding_smollm3_45

echo ""
echo "[4/8] SmolLM3 45-term behavioral (8 ck × 451 prompts)"
python src/eval_behavior_smollm3.py --all \
    --prompts data/prompts/canonical_45terms.jsonl \
    --outdir data/results/behavioral_smollm3_45

echo ""
echo "[5/8] C1-B + C4-B analysis — SmolLM3 45-term"
python src/analyze_c1b_within_term.py --model smollm345
python src/analyze_c4b_decoupling.py --model smollm345

# ── Step 3: C3 few-shot (OLMo and Qwen, sequential) ──────────────────────
echo ""
echo "[6/8] C3 few-shot — OLMo"
python src/run_c3_new_models.py --model olmo

echo ""
echo "C3 few-shot — Qwen"
python src/run_c3_new_models.py --model qwen

echo ""
echo "C3 few-shot — CRFM seed1"
python src/run_c3_new_models.py --model crfm

echo ""
echo "C3 few-shot — SmolLM3"
python src/run_c3_new_models.py --model smollm3

# ── Step 4: C5 causal ablation ────────────────────────────────────────────
echo ""
echo "[7/8] C5 causal ablation — OLMo"
python src/run_c5_new_models.py --model olmo

echo ""
echo "C5 causal ablation — Qwen"
python src/run_c5_new_models.py --model qwen

echo ""
echo "C5 causal ablation — CRFM"
python src/run_c5_new_models.py --model crfm

echo ""
echo "C5 causal ablation — SmolLM3"
python src/run_c5_new_models.py --model smollm3

# ── Step 5: Full matrix summary ───────────────────────────────────────────
echo ""
echo "[8/8] Full matrix summary"
python src/summarize_full_matrix.py

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE: $(date)"
echo "============================================================"
