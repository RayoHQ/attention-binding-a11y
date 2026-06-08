# Reproduction Checklist

## Environment Setup
- [ ] Python 3.9+ installed
- [ ] CUDA GPU available (see VRAM requirements below)
- [ ] **VRAM by model scale:**
  - [ ] 8GB: Pythia-160M, CRFM GPT-2 Small (117M parameters)
  - [ ] 12GB: Pythia-1B, OLMo-1B, Qwen2.5-1.5B
  - [ ] 16GB+: Pythia-2.8B, SmolLM3-3B (reduce batch size if needed)
- [ ] ~50GB free disk space for model checkpoints
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Run `python setup_data.py` to verify environment

## Data Preparation
- [ ] Verify `data/prompts/pilot_terms.jsonl` exists (12 prompts, 3 pilot terms)
- [ ] Verify expanded 9-term dataset files exist
- [ ] Verify 99-prompt robustness dataset exists
- [ ] Run `python src/tokenization_audit.py` to verify token spans

## Discriminant Validity (§4.1)
- [ ] Run V2 controls (true nonsense, cross-language, rare pairs)
- [ ] Verify gradient: nonsense (0.26) → rare pairs (0.50) → real terms (0.74), all p < 0.001
- [ ] Run V3 controls (domain-adjacent terms: 15 terms × 2 scales)
- [ ] Run V4 controls (wrong-domain terms: 18 terms × 2 scales)
- [ ] Verify scale-dependent discrimination: fails at 160M, partial at 1B
- [ ] Identify accidentally valid terms (heading tag, aria role, alt image)
- [ ] Generate discriminant validity figures

## C1: Lead-Lag Emergence (§4.3)
- [ ] Extract binding for 160M all checkpoints (0, 15k, 30k, 60k, 90k, 120k, 140k, 143k)
- [ ] Extract binding for 1B all checkpoints
- [ ] Extract binding for 2.8B all checkpoints
- [ ] Run behavioral evaluation for all 24 model-checkpoint combinations (9 terms)
- [ ] Compute correlations: early coupling (ρ = +0.57), late decoupling (ρ = −0.20)
- [ ] Verify per-term heterogeneity (aria attribute: ρ = +0.07, behavior 0.76)
- [ ] Generate Figure 1 (emergence curves) via `notebooks/figure1_emergence_curves.ipynb`
- [ ] Generate phase transition scatter plots

## C3: Few-Shot Unlockability (§4.5)
- [ ] Run pilot few-shot: `python src/eval_few_shot.py`
- [ ] Verify 160M step15k: zero-shot 0.333 → few-shot 0.944 (+61.1 pp)
- [ ] Verify 160M step30k: zero-shot 0.667 → few-shot 0.944 (+27.8 pp)
- [ ] Verify 1B step15k: zero-shot 0.556 → few-shot 0.944 (+38.9 pp)
- [ ] Run 99-prompt replication (9 terms × 6 generation prompts)
- [ ] Verify replication: ~30 pp improvement across all conditions
- [ ] Verify control (step 0): negligible few-shot improvement (EB* ≈ 0.15)

## Dataset Expansion & Robustness (§4.2)
- [ ] Run 99-prompt evaluation (9 terms × 11 prompts × 10 formats)
- [ ] Compute coefficient of variation (CV) per term: expected mean CV = 0.144
- [ ] Verify 7/9 terms have CV < 0.05 (stable)
- [ ] Verify 2/9 high-variance terms (aria attribute, landmark region)
- [ ] Test prompt length correlation: expected ρ = 0.036 (not significant)
- [ ] Generate prompt robustness heatmap

## Sampling Parameter Robustness (§4.2.3)
- [ ] Run generation with T = 0.0, 0.3, 0.7 (5 seeds each)
- [ ] Verify variability decreases: early (0.334) → late (0.211) at 2.8B
- [ ] Verify greedy decoding most stable at trained checkpoints

## C4: Scale-Dependent Decoupling (§4.4)
- [ ] Verify 1B EB* plateau at step 15k (0.646) vs behavior rise to 0.806
- [ ] Verify cross-scale comparison table (160M/1B/2.8B trajectories)
- [ ] Verify regression at convergence (160M: 0.667→0.500, 2.8B: 0.667→0.500)
- [ ] Generate Figure 4 (1B decoupling) via `notebooks/figure1_emergence_curves.ipynb`

## C5: Causal Ablation (§4.6)
- [ ] Run `python src/minimal_causal.py` (160M step 120K)
- [ ] Verify graded effects: top (−16.7 pp) > random (−6.7 pp) > bottom (0 pp)
- [ ] Run `python src/minimal_causal_28b.py` (2.8B step 143K)
- [ ] Verify reversal: top (+33.3 pp), random (0 pp), bottom (0 pp)
- [ ] Verify discriminant validity holds at both scales (top ≠ random = bottom)

## Final Verification
- [ ] All figures generated and match paper (Figures 1-9)
- [ ] All tables match paper values
- [ ] Run `python src/analysis_pilot.py` for summary statistics
- [ ] Verify 432 pilot observations (9 terms × 3 models × 8 checkpoints × 2)
- [ ] Verify 7-model cross-architecture results (41-term canonical register)
- [ ] (Optional) Run on clean environment to verify full reproducibility

## Resource Summary (Actual Experimental Scope)

**Checkpoint count:**
- Pythia: 3 models × 8 checkpoints = 24
- OLMo: 1 model × 8 checkpoints = 8
- CRFM: 5 seeds × 8 checkpoints = 40
- SmolLM3: 1 model × 8 checkpoints = 8
- Qwen: 1 model × 1 checkpoint = 1
- **Total: 81 checkpoints × 41 terms × 205 prompts = 681,105 inference evaluations**

**Minimum (Pythia-only pilot, 9 terms):**
- VRAM: 8GB (160M), 12GB (1B), 16GB+ (2.8B)
- Time: ~3–5 hours GPU / ~8–12 hours CPU
- Storage: ~15GB

**Full reproduction (7 models, 41 terms, all claims):**
- VRAM: 16GB+ recommended (SmolLM3-3B requires most memory)
- Time: ~20–25 hours GPU / ~50–70 hours CPU
- Storage: ~50–70GB

**Total project effort (Feb 6–Apr 20, incl. R&D):**
- Pilot phase (Feb 6–8, C3/C5 initial validation): ~5–10 hours GPU
- Expansion phase (Apr 3–5, 100-prompt + discriminant validity + robustness): ~5–10 hours GPU
- Cross-architecture wave (Apr 13–20, 41-term CRFM/SmolLM3/OLMo/Qwen + multi-seed): ~15–20 hours GPU
- Failed runs, debugging, analysis iterations: ~5–10 hours GPU
- **Total R&D effort: ~40–60 hours GPU / ~100–140 hours CPU**

**Per-checkpoint binding extraction (reference, includes model load):**
- 160M / CRFM: ~1–2 min
- 1B / OLMo: ~2–4 min
- 2.8B: ~8–12 min
- SmolLM3-3B: ~1–2 min

**Per-checkpoint behavioral eval (205 prompts, includes model load):**
- 160M / CRFM: ~1–2 min
- 1B / OLMo: ~2–4 min
- 2.8B: ~8–12 min
- SmolLM3-3B: ~2–5 min

## New Figures & Tables (Post-April 3)
- [ ] Figure: discriminant_validity_controls.pdf (V2/V3/V4 gradient)
- [ ] Figure: prompt_robustness_heatmap.pdf (99-prompt CV analysis)
- [ ] Figure: phase_transition_scatter.pdf (early vs late correlation patterns)
- [ ] Figure: term_heterogeneity_2b8.pdf (9-term trajectories at 2.8B)
- [ ] Table A.1b: Per-term performance (9 terms)
- [ ] Table A.1c: V1 (failed) and V2 (successful) controls
- [ ] Table A.1d: V3 domain-adjacent terms
- [ ] Table A.1e: V4 wrong-domain terms
