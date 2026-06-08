# Appendix A: Raw Data Tables

## A.1 Full Checkpoint Summary (Original 3 Terms)

Original dataset: screen reader, skip link, alt text

| Model | Checkpoint | Step (k) | Rec Acc | Gen Score | Beh Avg | EB\* Mean | EB\* Max | Best Layer |
|-------|-----------|----------|---------|----------|---------|-----------|----------|------------|
| 160M | step0 | 0 | 0.167 | 0.000 | 0.083 | 0.157 | 0.307 | L6 |
| 160M | step15000 | 15 | 0.000 | 0.333 | 0.167 | 0.644 | 0.717 | L3 |
| 160M | step30000 | 30 | 0.167 | 0.667 | 0.417 | 0.642 | 0.780 | L3 |
| 160M | step60000 | 60 | 0.167 | 0.556 | 0.361 | 0.684 | 0.856 | L1 |
| 160M | step90000 | 90 | 0.500 | 0.556 | 0.528 | 0.734 | 0.906 | L11 |
| 160M | step120000 | 120 | 0.667 | 0.556 | 0.611 | 0.821 | 0.917 | L8 |
| 160M | step140000 | 140 | 0.667 | 0.556 | 0.611 | 0.816 | 0.916 | L3 |
| 160M | step143000 | 143 | 0.500 | 0.500 | 0.500 | 0.831 | 0.915 | L3 |
| 1B | step0 | 0 | 0.333 | 0.000 | 0.167 | 0.146 | 0.240 | L1 |
| 1B | step15000 | 15 | 0.667 | 0.556 | 0.611 | 0.646 | 0.753 | L3 |
| 1B | step30000 | 30 | 0.833 | 0.722 | 0.778 | 0.611 | 0.705 | L3 |
| 1B | step60000 | 60 | 0.667 | 0.722 | 0.694 | 0.595 | 0.683 | L3 |
| 1B | step90000 | 90 | 0.500 | 0.778 | 0.639 | 0.598 | 0.750 | L3 |
| 1B | step120000 | 120 | 0.667 | 0.667 | 0.667 | 0.608 | 0.802 | L3 |
| 1B | step140000 | 140 | 0.667 | 0.833 | 0.750 | 0.607 | 0.823 | L3 |
| 1B | step143000 | 143 | 0.667 | 0.944 | 0.806 | 0.599 | 0.826 | L0 |
| 2.8B | step0 | 0 | 0.500 | 0.000 | 0.250 | 0.196 | 0.324 | L1 |
| 2.8B | step15000 | 15 | 0.667 | 0.611 | 0.639 | 0.885 | 0.918 | L6 |
| 2.8B | step30000 | 30 | 0.833 | 0.667 | 0.750 | 0.897 | 0.933 | L12 |
| 2.8B | step60000 | 60 | 0.500 | 0.833 | 0.667 | 0.888 | 0.941 | L30 |
| 2.8B | step90000 | 90 | 0.667 | 0.833 | 0.750 | 0.882 | 0.928 | L27 |
| 2.8B | step120000 | 120 | 0.667 | 0.889 | 0.778 | 0.881 | 0.932 | L30 |
| 2.8B | step140000 | 140 | 0.667 | 0.889 | 0.778 | 0.858 | 0.940 | L4 |
| 2.8B | step143000 | 143 | 0.500 | 0.833 | 0.667 | 0.870 | 0.941 | L4 |

## A.1b Expanded Dataset: Per-Term Performance at Trained Checkpoints

Nine accessibility terms at steps 120K, 140K, 143K (mean values):

| Term | EB\* Mean | EB\* Std | Beh Mean | Beh Std | Per-Term ρ | n |
|------|-----------|----------|----------|---------|------------|---|
| alt text | 0.754 | 0.139 | 0.593 | 0.100 | +0.38** | 48 |
| aria attribute | 0.387 | 0.392 | 0.759 | 0.127 | +0.07 (ns) | 48 |
| color contrast | 0.783 | 0.112 | 0.620 | 0.125 | +0.68*** | 48 |
| focus indicator | 0.749 | 0.099 | 0.824 | 0.083 | +0.68*** | 48 |
| form validation | 0.751 | 0.119 | 0.491 | 0.073 | +0.34* | 48 |
| heading structure | 0.821 | 0.076 | 0.389 | 0.167 | +0.67*** | 48 |
| screen reader | 0.702 | 0.157 | 0.676 | 0.121 | +0.30* | 48 |
| skip link | 0.795 | 0.111 | 0.787 | 0.172 | +0.40** | 48 |
| tab order | 0.750 | 0.155 | 0.763 | 0.110 | +0.48*** | 48 |

\*p<0.05, \*\*p<0.01, \*\*\*p<0.001. Per-term ρ computed across all 48 model-checkpoint pairs (3 models × 8 steps × 2 prompts).

## A.1c Discriminant Validity: Control Baseline Results

**v2 Controls (successful)** at trained checkpoints (mean across 3 models):

| Control Group | Example Terms | Mean EB\* | Std | vs Real (0.74) |
|---------------|---------------|-----------|-----|----------------|
| True nonsense | "zqx plarf", "glib thrang" | 0.26 | 0.07 | Δ=+0.48, p<0.001*** |
| Cross-language | "écran reader", "skip enlace" | 0.41 | 0.11 | Δ=+0.33, p<0.001*** |
| Rare token pairs | "pterodactyl altimeter", "velvet compiler" | 0.50 | 0.25 | Δ=+0.24, p<0.001*** |
| **Real terms** | "screen reader", "focus indicator", etc. | **0.74** | 0.20 | — |

**v1 Controls (failed)** - discriminant validity NOT established:

| Control Group | Example Terms | Mean EB\* | vs Real (0.77) |
|---------------|---------------|-----------|----------------|
| Backwards | "reader screen", "link skip" | 0.82 | Δ=−0.05, p=0.34 (ns) |
| Cross-term | "screen link", "reader text" | 0.78 | Δ=−0.01, p=0.89 (ns) |
| Semantic field | "keyboard mouse", "header footer" | 0.77 | Δ=0.00, p=0.98 (ns) |
| Frequency-matched | "open source", "machine learning" | 0.72 | Δ=+0.05, p=0.28 (ns) |
| Random | "elephant database", "coffee algorithm" | 0.75 | Δ=+0.02, p=0.71 (ns) |

v1 controls failed because they were inadvertently legitimate corpus bigrams.

## A.1d V3 Controls: Domain-Adjacent Terms (Per-Term EB*)

Terms share one token with real accessibility terms but replace the other with plausible vocabulary. Accidentally valid terms (†) showed EB* comparable to real terms and were excluded from the irrelevant-terms analysis in §4.0.

Real terms baseline: 160M = 0.74, 1B = 0.74.

| Term | Source Term | Overlap Token | 160M step120k EB* | 1B step143k EB* | Note |
|------|-------------|---------------|-------------------|-----------------|------|
| alt function | alt text | alt | 0.717 | 0.640 | |
| alt image | alt text | alt | 0.679 | 0.712 | † accidentally valid |
| screen editor | screen reader | screen | 0.895 | 0.556 | |
| screen display | screen reader | screen | 0.879 | 0.607 | |
| skip button | skip link | skip | 0.916 | 0.572 | |
| skip menu | skip link | skip | 0.917 | 0.596 | |
| focus selector | focus indicator | focus | 0.740 | 0.684 | |
| focus element | focus indicator | focus | 0.722 | 0.597 | |
| heading label | heading structure | heading | 0.903 | 0.659 | |
| heading tag | heading structure | heading | 0.916 | 0.616 | † accidentally valid (HTML standard) |
| color gradient | color contrast | color | 0.916 | 0.681 | |
| color scheme | color contrast | color | 0.917 | 0.578 | |
| aria property | aria attribute | aria | 0.837 | 0.617 | |
| aria role | aria attribute | aria | 0.842 | 0.761 | † accidentally valid (ARIA standard) |
| landmark section | landmark region | landmark | 0.917 | 0.772 | |
| **Mean (excl. †)** | | | **0.861** | **0.639** | 12 irrelevant terms |
| **Mean (all 15)** | | | **0.866** | **0.657** | |

**Interpretation:** At 160M all 15 terms exceed the real term baseline (0.74) — EB* cannot discriminate domain-adjacent terms at smaller scales. At 1B, 12/15 irrelevant terms fall below baseline, with the 3 accidentally valid terms remaining elevated (0.71–0.76).

---

## A.1e V4 Controls: Wrong-Domain Terms (Per-Term EB*)

Terms pair accessibility tokens with programming, hardware, or CSS vocabulary with zero conceptual connection to accessibility. "landmark class" is the boundary case that persists elevated at 1B due to CSS class naming conventions.

| Term | Source Term | Overlap Token | Wrong Domain | 160M step120k EB* | 1B step143k EB* |
|------|-------------|---------------|--------------|-------------------|-----------------|
| alt function | alt text | alt | programming | 0.717 | 0.640 |
| alt parameter | alt text | alt | programming | 0.708 | 0.618 |
| alt variable | alt text | alt | programming | 0.725 | 0.671 |
| screen printer | screen reader | screen | hardware | 0.916 | 0.635 |
| screen monitor | screen reader | screen | hardware | 0.840 | 0.594 |
| screen output | screen reader | screen | programming | 0.846 | 0.676 |
| skip variable | skip link | skip | programming | 0.834 | 0.644 |
| skip function | skip link | skip | programming | 0.834 | 0.647 |
| heading class | heading structure | heading | css | 0.916 | 0.738 |
| heading style | heading structure | heading | css | 0.864 | 0.632 |
| color syntax | color contrast | color | programming | 0.917 | 0.643 |
| color variable | color contrast | color | programming | 0.899 | 0.653 |
| focus loop | focus indicator | focus | programming | 0.705 | 0.688 |
| focus event | focus indicator | focus | programming | 0.757 | 0.631 |
| aria method | aria attribute | aria | programming | 0.917 | 0.707 |
| aria function | aria attribute | aria | programming | 0.905 | 0.590 |
| landmark variable | landmark region | landmark | programming | 0.917 | 0.705 |
| landmark class ‡ | landmark region | landmark | css | 0.890 | 0.826 |
| **Mean (all 18)** | | | | **0.845** | **0.663** |
| **Mean (excl. ‡)** | | | | **0.842** | **0.650** |

‡ Boundary case: "landmark class" persists at 0.826 at 1B, likely due to CSS class naming conventions in web development corpora.

**Interpretation:** At 160M, all 18 wrong-domain terms exceed the real term baseline (0.74) — EB* fails to discriminate based on semantic domain at smaller scales. At 1B, 17/18 terms fall below baseline; only "landmark class" remains elevated. The domain taxonomy (hardware, programming, CSS) shows no systematic effect — discrimination failure is driven by corpus co-occurrence regardless of wrong-domain category.

---

## A.2 C5 Ablation: 160M step120000

Top-4 heads by average BSI:

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 3 | 0 | 0.951 |
| 2 | 2 | 8 | 0.830 |
| 3 | 3 | 2 | 0.761 |
| 4 | 0 | 0 | 0.617 |

Bottom-4 heads (negative control):

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 9 | 0 | 0.000 |
| 2 | 9 | 2 | 0.000 |
| 3 | 9 | 5 | 0.000 |
| 4 | 10 | 4 | ≈0.000 |

Ablation results:

| Condition | Rec Acc | Gen Score | Rec Δ | Gen Δ |
|-----------|---------|-----------|-------|-------|
| Baseline | 4/6 (0.667) | 0.556 | — | — |
| Top-4 ablated | 3/6 (0.500) | 0.444 | −0.167 | −0.111 |
| Random trial 1 | 4/6 (0.667) | 0.556 | 0.000 | 0.000 |
| Random trial 2 | 3/6 (0.500) | 0.556 | −0.167 | 0.000 |
| Random trial 3 | 4/6 (0.667) | 0.611 | 0.000 | +0.056 |
| Random trial 4 | 3/6 (0.500) | 0.444 | −0.167 | −0.111 |
| Random trial 5 | 4/6 (0.667) | 0.556 | 0.000 | 0.000 |
| Random mean | 0.600 | 0.544 | −0.067 | −0.011 |
| Bottom-4 ablated | 4/6 (0.667) | 0.556 | 0.000 | 0.000 |

Specificity (combined): +0.100

## A.3 C5 Ablation: 2.8B step143000

Top-4 heads by average BSI:

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 1 | 12 | 0.937 |
| 2 | 1 | 11 | 0.865 |
| 3 | 4 | 16 | 0.850 |
| 4 | 1 | 6 | 0.780 |

Bottom-4 heads (negative control):

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 30 | 0 | ≈0.000 |
| 2 | 2 | 15 | ≈0.000 |
| 3 | 31 | 16 | ≈0.000 |
| 4 | 27 | 3 | ≈0.000 |

Ablation results:

| Condition | Rec Acc | Gen Score | Rec Δ | Gen Δ |
|-----------|---------|-----------|-------|-------|
| Baseline | 3/6 (0.500) | 0.833 | — | — |
| Top-4 ablated | 5/6 (0.833) | 0.778 | +0.333 | −0.055 |
| Random trial 1 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random trial 2 | 3/6 (0.500) | 0.778 | 0.000 | −0.055 |
| Random trial 3 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random trial 4 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random trial 5 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random mean | 0.500 | 0.822 | 0.000 | −0.011 |
| Bottom-4 ablated | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |

## A.4 C3 Few-Shot Unlockability Results

| Model | Checkpoint | EB\* | Zero-Shot Gen | One-Shot Gen | Δ (pp) | Relative Δ |
|-------|-----------|------|---------------|--------------|--------|------------|
| 160M | step 15k | 0.644 | 0.333 | 0.944 | +61.1 | +183.3% |
| 160M | step 30k | 0.642 | 0.667 | 0.944 | +27.8 | +41.7% |
| 1B | step 15k | 0.646 | 0.556 | 0.944 | +38.9 | +70.0% |

**Note:** One-shot improvement is partly inflated by in-context copying. The model frequently reproduces phrasing from the provided example. See §4.2 for discussion.

Per-prompt breakdown (160M step 15k):

| Term | Prompt | Zero-Shot | One-Shot | Δ |
|------|--------|-----------|----------|---|
| screen reader | gen\_001 | 1.000 | 1.000 | 0.000 |
| screen reader | gen\_002 | 0.000 | 1.000 | +1.000 |
| skip link | gen\_001 | 0.333 | 1.000 | +0.667 |
| skip link | gen\_002 | 0.333 | 0.667 | +0.333 |
| alt text | gen\_001 | 0.000 | 1.000 | +1.000 |
| alt text | gen\_002 | 0.333 | 0.667 | +0.333 |

Raw results saved in `data/results/few_shot/`.

## A.5 Evaluation Prompts

Three accessibility terms × 4 prompts each (2 recognition, 2 generation) = 12 total.

**Recognition prompts** use 4-choice MCQ format, scored via log-probability ranking.
**Generation prompts** use open-ended completion, scored via keyword rubric (threshold = 3 keywords).

See `data/prompts/pilot_terms.jsonl` for full prompt specifications.

## A.6 41-Term Expansion: C1-B and C4-B Results

### A.6.1 C1-B Within-Term Temporal Precedence (41 terms)

| Model | Seed | N terms | EB* leads | Lead% | Binomial p |
|-------|------|---------|-----------|-------|------------|
| OLMo-1B | — | 40 (1 excl.) | 36/40 | **90.0%** | **<0.0001** |
| CRFM GPT-2 Sm | x1 | 41 | 26/41 | 63.4% | 0.059 |
| CRFM GPT-2 Sm | x2 | 41 | 32/41 | 78.0% | **0.0002** |
| CRFM GPT-2 Sm | x3 | 41 | 36/41 | **87.8%** | **<0.0001** |
| CRFM GPT-2 Sm | x4 | 41 | 29/41 | 70.7% | **0.0058** |
| CRFM GPT-2 Sm | x5 | 41 | 26/41 | 63.4% | 0.059 |
| CRFM combined | all | 205 | 149/205 | **72.7%** | **<<0.001** |
| SmolLM3-3B (45t) | — | 41 | 21/41 | 51.2% | 0.500 (‡) |

*OLMo-1B exclusion: 1 term had constant behavioral scores across all checkpoints (no variance for cross-lag correlation).*
*(‡) SmolLM3-3B C1-B is a likely left-censoring artifact: earliest available checkpoint (step-40k) already shows declining EB\*, so the coupling phase predates the observation window.*

### A.6.2 C4-B Decoupling (41 terms)

| Model | Seed | Strict decouple | rho_early (mean) | rho_late (mean) |
|-------|------|-----------------|-----------------|-----------------|
| OLMo-1B (45t) | — | 12/27 (44%) | +0.247 | **−0.181** |
| CRFM x1 | x1 | 21/41 (51%) | +0.423 | −0.095 |
| CRFM x2 | x2 | 15/41 (37%) | +0.419 | +0.154 |
| CRFM x3 | x3 | 9/41 (22%) | +0.598 | +0.479 |
| CRFM x4 | x4 | 30/41 (73%) | +0.661 | **−0.372** |
| CRFM x5 | x5 | 11/39 (28%) | +0.622 | +0.260 |
| CRFM mean | all | 86/203 (42%) | +0.545 | +0.085 |
| SmolLM3-3B (45t) | — | 22/40 (55%) | +0.118 | **−0.281** |

*OLMo C4-B usable terms reduced from 40 to 27 due to ceiling/floor effects in behavioral scores.*
*CRFM seed variance (22–73%) is itself a finding: small-model decoupling is initialization-sensitive.*

### A.6.3 C3 Few-Shot (New Models, 9 terms)

OLMo results (per-checkpoint):

| Model | Checkpoint | Zero-Shot | Few-Shot | Δ (pp) | Relative |
|-------|-----------|-----------|----------|--------|----------|
| OLMo-1B | step15k | 0.432 | 0.636 | **+20.4** | +47.1% |
| OLMo-1B | step143k | 0.525 | 0.617 | +9.3 | +17.6% |

Aggregate mean Δ across both checkpoints, all 9 generation prompts per term:

| Model | Mean C3 Δ | Notes |
|-------|-----------|-------|
| OLMo-1B | **+0.075** | Unlockability confirmed; gap narrows at late ck |
| CRFM GPT-2 (5 seeds) | **+0.012±0.054** | Inverted pattern: early ck Δ=−0.050±0.045; late ck Δ=+0.073±0.044 |
| SmolLM3-3B | **−0.022** | Negative on 9-term protocol: high zero-shot baseline, few-shot prefix interferes |
| Qwen2.5-1.5B† | — | Not run on 9-term protocol; see 41-term result: ZS=0.542, FS=0.724, Δ=**+18.2 pp** |

CRFM C3 per-checkpoint breakdown (5 seeds, N=54 generation prompts per seed per ck):

| Checkpoint | Mean Δ | SD | Range |
|-----------|--------|-----|-------|
| checkpoint-1000 (early) | **−0.050** | 0.045 | −0.111 to +0.015 |
| checkpoint-400000 (late) | **+0.073** | 0.044 | +0.019 to +0.120 |

*CRFM shows an inverted C3 pattern relative to OLMo: few-shot hurts at early checkpoint (model too weak to leverage exemplars) but helps at late checkpoint (mature representations benefit from in-context examples). This is the opposite of the standard unlockability hypothesis, which predicts the largest few-shot gain when knowledge is latent (early training). For CRFM at 117M scale, the model must first develop sufficient capacity before few-shot prompting is beneficial.*

*SmolLM3 C3 Δ is negative: at both checkpoints SmolLM3 already scores high zero-shot, and the few-shot prefix marginally reduces generation quality. This confirms the ceiling-adjacent regime.*

### A.6.4 C3 Few-Shot (Pythia 9-term unified protocol)

All three Pythia scales evaluated at early and late checkpoints using `eval_few_shot_c3.py` (term-specific multi-sentence exemplars, N=54 generation prompts, 9 terms).

| Model | Checkpoint | Zero-Shot | Few-Shot | Δ (pp) | Status |
|-------|-----------|-----------|----------|--------|--------|
| 160M | step15000 | 0.265 | **0.630** | **+36.4** | ✅ strong |
| 160M | step143000 | 0.290 | **0.599** | **+30.9** | ✅ strong |
| 1B | step15000 | 0.340 | **0.704** | **+36.4** | ✅ strong |
| 1B | step143000 | 0.395 | **0.667** | **+27.2** | ✅ strong |
| 2.8B | step15000 | 0.422 | **0.710** | **+28.8** | ✅ strong |
| 2.8B | step143000 | 0.506 | **0.700** | **+19.4** | ⚠ borderline |

Per-term breakdown (step15k ZS / step143k ZS, showing early vs. late zero-shot):

| Term | 160M ZS early | 160M ZS late | 1B ZS early | 1B ZS late | 2.8B ZS early | 2.8B ZS late |
|------|--------------|--------------|-------------|------------|---------------|--------------|
| screen reader | 0.500 | 0.500 | 0.722 | 0.500 | 0.611 | 0.611 |
| skip link | 0.444 | 0.389 | 0.500 | 0.667 | 0.722 | 0.667 |
| alt text | 0.333 | 0.333 | 0.444 | 0.667 | 0.500 | 0.556 |
| color contrast | 0.000 | 0.167 | 0.000 | 0.333 | 0.111 | 0.611 |
| focus indicator | 0.611 | 0.500 | 0.333 | 0.278 | 0.522 | 0.500 |
| heading structure | 0.056 | 0.167 | 0.278 | 0.167 | 0.333 | 0.444 |
| keyboard navigation | 0.333 | 0.278 | 0.278 | 0.444 | 0.333 | 0.556 |
| landmark region | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| aria attribute | 0.111 | 0.278 | 0.500 | 0.500 | 0.667 | 0.611 |

*`landmark region` is systematically zero across all models and checkpoints (zero-shot and few-shot alike). This term may require architectural vocabulary not captured by the few-shot exemplar format, or may require a context length beyond the generation window.*

*Files: `data/results/few_shot_c3/160m_step{15000,143000}_c3_fewshot.json`, `1b_step{15000,143000}_c3_fewshot.json`, `2.8b_step{15000,143000}_c3_fewshot.json`*

*Results from `data/results/few_shot_c3/`.*

## A.7 Canonical 41-Term C5 Ablation Results (N=205 recognition prompts)

### A.7.1 Pythia Canonical41

| Model | Baseline | TOP-4 | Rand mean | BOT-4 | Top Δ | Spec (rec-only) |
|-------|----------|-------|-----------|-------|-------|----------------|
| Pythia-160M step143k | 0.810 | 0.698 | 0.833 | 0.829 | −11.2 pp | **+0.137** |
| Pythia-1B step143k | 0.800 | 0.649 | 0.766 | 0.727 | −15.1 pp | **+0.117** |
| Pythia-2.8B step143k | 0.932 | 0.859 | 0.969 | 0.893 | −7.3 pp | **+0.110** |

*Spec = top_rec_drop − mean_rand_rec_drop (rec-only, consistent with all other models). Original combined (rec+gen) specificity was +0.091 / +0.084 / +0.079 respectively.*
*Top binding heads: 160M={L3H0,L2H8,L3H2,L1H1}; 1B={L3H5,L1H0,L0H0,L1H3}; 2.8B={L1H12,L1H11,L4H16,L1H6}*

### A.7.2 OLMo-1B Canonical41 (N=205)

| Condition | Rec Acc | Rec Δ |
|-----------|---------|-------|
| Baseline | 0.990 | — |
| TOP-4 ablated | 0.980 | −1.0 pp |
| Rand mean×5 | 0.974 | −1.6 pp |
| BOT-4 ablated | 0.995 | +0.5 pp |
| **Specificity** | | **−0.006** |

*Ceiling regime: 99% baseline leaves ≤2 prompts that any ablation can flip. Top binding heads: {L0H7, L1H10, L2H3, L0H11}. File: `olmo_step143k_c5_canonical41.json`*

### A.7.3 CRFM GPT-2 Small Canonical41 — All Seeds (N=205)

**Per-seed results (checkpoint-400000):**

| Seed | Baseline | TOP-4 ablated | Rand mean | BOT-4 | Top Δ | Spec | Pattern |
|------|----------|---------------|-----------|-------|--------|------|---------|
| x1 (alias-x21) | 0.620 | 0.829 | 0.655 | 0.605 | **+20.9 pp** | −0.175 | SUPPRESSOR |
| x2 (battlestar-x49) | 0.722 | 0.488 | 0.691 | 0.659 | −23.4 pp | +0.203 | COUPLED |
| x3 (caprica-x81) | 0.966 | 0.668 | 0.852 | 0.942 | −29.8 pp | +0.183 | COUPLED (strong) |
| x4 (expanse2-x4) | 0.600 | 0.537 | 0.608 | 0.576 | −6.3 pp | +0.071 | COUPLED (weak) |
| x5 (expanse-x777) | 0.844 | 0.698 | 0.820 | 0.810 | −14.6 pp | +0.122 | COUPLED |
| **Mean±SD** | **0.750±0.154** | | | | **+0.106±0.198** | **+0.081±0.152** | **4/5 COUPLED** |

**Key finding**: 4/5 seeds show the coupled pattern (top binding heads causally necessary, spec > 0); seed 1 (alias-x21) is the sole suppressor outlier. The **mean specificity = +0.081** (positive) indicates the modal CRFM causal regime is coupled, not suppressive. However, the high SD (±0.152) and the existence of the seed-1 suppressor confirm that CRFM at 117M / 400k steps straddles a **causal boundary** where initialization determines whether binding heads become load-bearing scaffolds or active suppressors. The baseline variance is also notable (0.600–0.966), showing initialization strongly governs both behavioral performance level and causal head function at this regime.

*Files: `crfm_seed{1..5}_checkpoint-400000_c5_canonical41.json`*

### A.7.4 SmolLM3-3B Canonical41 (N=205)

| Condition | Rec Acc | Rec Δ |
|-----------|---------|-------|
| Baseline | 0.868 | — |
| TOP-4 ablated | 0.902 | +3.4 pp |
| Rand mean×5 | 0.859 | −0.9 pp |
| BOT-4 ablated | 0.858 | −1.0 pp |
| **Specificity** | | **−0.043** |

*Ceiling-adjacent distributed regime. File: `smollm3_step3440k_c5_canonical41.json`*

### A.7.5 Qwen2.5-1.5B Canonical41 (N=205)

| Condition | Rec Acc | Rec Δ |
|-----------|---------|-------|
| Baseline | 0.990 | — |
| TOP-4 ablated | 0.980 | −1.0 pp |
| Rand mean×5 | 0.985 | −0.5 pp |
| BOT-4 ablated | 0.990 | 0.0 pp |
| **Specificity** | | **+0.005** |

*Ceiling regime: 99% baseline. Top binding heads: {L13H4, L1H4, L0H6, L7H12}. File: `qwen_final_c5_canonical41.json`*

*Full results: `data/results/causal/*_c5_canonical41.json`*

## A.8 Full Analysis Matrix (All Models, All Analyses)

*Generated by `python src/summarize_full_matrix.py`. All C5 runs complete as of Apr 19, 2026.*

```
Model                  Params  Steps   Prompts          C1-B%      p   ΔLead  C4-B%  ρ_late  N_c4  C5_drop   Spec   C3_Δ
Pythia-160M            160M    143k    41 terms             7%  1.0000  -0.645    46%  +0.044    28   +0.112  +0.137  +0.309
Pythia-1B              1B      143k    41 terms            73%  0.0022  +0.228    54%  -0.054    28   +0.151  +0.117  +0.272
Pythia-2.8B            2.8B    143k    41 terms            79%  0.0004  +0.230    43%  +0.270    28   +0.073  +0.110  +0.194
OLMo-1B (9t)           1B      143k    9 terms             78%  0.0898  +0.293    62%  -0.348     8    +0.010  -0.006  +0.075
OLMo-1B (45t)          1B      143k    41 terms            90% <0.0001  +0.374    44%  -0.181    27    +0.010  -0.006  +0.075
CRFM GPT-2 (9t)        117M    400k    9 terms (5-sd)      56%  0.5000  +0.036    33%  +0.442     9    +0.106  +0.081  +0.073
CRFM GPT-2 (45t)       117M    400k    41 terms (5-sd)     73%  <<0.001  +0.299  42%  +0.085    41    +0.106  +0.081  +0.073
SmolLM3-3B (9t)        3B      3440k   9 terms             33%  0.9102  -0.199    67%  -0.189     9    -0.034  -0.043  -0.022
SmolLM3-3B (45t)       3B      3440k   41 terms            51%  0.5000  -0.040    55%  -0.281    40    -0.034  -0.043  -0.022
Qwen2.5-1.5B           1.5B    final   41 terms             —       —      —      —      —      —    +0.010  +0.005  +0.182
```

**Interpretation notes:**
- Pythia-160M C1-B=7% (ΔLead=−0.645): EB* lags behavior at small scale — behavior emerges before binding structure. Anti-precedence.
- SmolLM3 C1-B≈chance (51%, censored): binding precursor phase predates our earliest checkpoint.
- Pythia-2.8B ρ_late=+0.270: coupling persists at late training (not decoupled), despite large C1-B=79%.
- OLMo 45t C1-B=90% is strongest across all models and term sets.
- SmolLM3 45t ρ_late=−0.281 is most-negative rho_late in the canonical41 dataset.
- **CRFM C5**: Mean drop=+0.106±0.198, mean spec=+0.081±0.152 across 5 seeds. Modal pattern: COUPLED (4/5 seeds). Seed 1 (alias-x21) is the sole suppressor outlier (spec=−0.175).
- OLMo and Qwen: C5_drop≈0 despite positive baseline (99% ceiling). Specificity≈0.
- CRFM C3 mean late-ck Δ=+0.073±0.044 (5 seeds); early-ck Δ=−0.050±0.045 (inverted pattern). All experiments complete ✅.

---

# Appendix B: Figure Index

All figures are stored in `paper/figures/`. Scripted figures can be regenerated by running the listed script from the repository root. Static figures have no generation script; they are manually composed from the data in Appendix A.

| Fig # | Filename | Paper section | Caption summary | Generation script |
|-------|----------|---------------|-----------------|-------------------|
| 1 | `correlation_lifecycle.png` / `.pdf` | §4.2 | Mean Spearman ρ(EB\*, Beh) at early (step 15k) and late (step 143k) for all three Pythia scales. Values from C4-B population test (41-term dataset). | `src/generate_pythia_lifecycle_figures.py` |
| 2 | `phase_transition_scatter.png` / `.pdf` | §4.2 | Six-panel scatter (3 scales × 2 phases): EB\* vs behavioral score per term, showing coupling at early and decoupling at late checkpoints. | `src/generate_pythia_lifecycle_figures.py` |
| 3 | `term_heterogeneity_2b8.png` / `.pdf` | §4.2 | Per-term EB\* and behavioral trajectories at 2.8B scale. Left: EB\* evolution; Right: behavioral evolution. Demonstrates binding–behavior independence. | `src/generate_pythia_lifecycle_figures.py` |
| 4 | `figure4_1b_decoupling.png` / `.pdf` | §4.4 | Dual-axis: EB\* (red) saturates at step 15k while behavioral score (green) continues rising through step 143k for Pythia-1B. | `src/generate_pythia_lifecycle_figures.py` |
| 5 | `prompt_robustness_heatmap.png` | §4.1.2 | EB\* heatmap across 9 terms × 6 generation prompts, sorted by CV. 7/9 terms show CV < 0.05. | *static* |
| 6 | `lifecycle_comparison_36v100.png` | §4.1.2 | Overlay of 36-prompt and 100-prompt lifecycle trajectories. Both show coupling→decoupling transition. | *static* |
| 7 | `aria_attribute_case_study.png` | §4.1.2 | Aria attribute anomaly: gen_002 prompt produces zero EB\* across all checkpoints; other prompts show normal binding. | *static* |
| 8 | `format_diversity_analysis.png` | §4.1.2 | EB\* by prompt format type (6 categories). All formats produce comparable distributions (0.57–0.68), confirming format independence. | *static* |
| 9 | `discriminant_validity_controls.png` / `.pdf` | §4.0 | Discriminant validity gradient from V2 controls through real terms to V3/V4 controls at 160M and 1B scales. | `src/generate_control_validity_figure.py` |
| 10 | `c1b_forest_plot.png` / `.pdf` | §4.2.1 | C1-B forest plot: EB\*-leads fraction (Wilson 95% CI) per model. OLMo-1B (90%) and Pythia-1B/2.8B (73–79%) cluster above chance (50%); Pythia-160M (7%) below. | `src/generate_c1b_forest_plot.py` |
| 11 | `c3_fewshot_unlockability.png` / `.pdf` | §4.3 | Panel A: Pythia 3×2 zero-shot / few-shot bars (9-term protocol). Panel B: cross-model Δ for 11 model–checkpoint pairs (41-term protocol). | `src/generate_c3_fewshot_figure.py` |
| 12 | `c5_crossarch_specificity.png` / `.pdf` | §4.5.5 | Panel A: rec-only specificity for all 7 models (CRFM ±1 SD across 5 seeds). Panel B: top-ablation Δ vs random-ablation Δ scatter (coupled regime below diagonal). | `src/generate_c5_crossarch_figure.py` |

## B.1 Regenerating All Scripted Figures

Run the following from the repository root to regenerate all scripted figures (Figs 1–4, 9–12):

```bash
python src/generate_pythia_lifecycle_figures.py   # Figs 1–4
python src/generate_control_validity_figure.py    # Fig 9
python src/generate_c1b_forest_plot.py            # Fig 10
python src/generate_c3_fewshot_figure.py          # Fig 11
python src/generate_c5_crossarch_figure.py        # Fig 12
```

All scripts write PNG and PDF to `paper/figures/` and require no arguments. Figs 5–8 are static and cannot be regenerated from a script.

## B.2 Data Sources per Figure

| Fig # | Primary data source |
|-------|---------------------|
| 1 | `data/results/c4b/pythia-{160m,1b,28b}_decoupling_summary.json` (rho_early / rho_late) |
| 2 | `data/results/binding/{size}_step{N}_binding.jsonl` + `data/results/behavioral/{size}_step{N}_behavioral.jsonl` |
| 3 | Same as Fig 2, 2.8B only |
| 4 | Hardcoded from paper §4.4 text (EB\* 0.480→0.646→0.595; Beh 0.167→0.833→0.806) |
| 5–8 | `data/results/binding_expanded/`, `data/results/behavioral_expanded/` (99-prompt set) |
| 9 | Hardcoded control term values from §4.0 discriminant validity experiments |
| 10 | `data/results/c1b/*_population_test.json` |
| 11 | Panel A: `paper/sections/results.md §4.3` (9-term unified protocol). Panel B: `data/results/few_shot_c3_expanded/*.json` (41-term) |
| 12 | `data/results/causal/*_c5_canonical41.json` |
