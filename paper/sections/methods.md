# 3. Methods

## 3.1 Models and Training Checkpoints

We use the Pythia model suite (Biderman et al., 2023), a family of autoregressive language models trained on the Pile dataset (Gao et al., 2020) with publicly available intermediate checkpoints. We study three model scales:

| Model | Parameters | Layers | Heads | Head Dim | Total Heads |
|-------|-----------|--------|-------|----------|-------------|
| Pythia-160M-deduped | 160M | 12 | 12 | 64 | 144 |
| Pythia-1B-deduped | 1B | 16 | 8 | 128 | 128 |
| Pythia-2.8B-deduped | 2.8B | 32 | 32 | 80 | 1,024 |

For each model, we evaluate eight checkpoints spanning the full training trajectory: step 0, 15k, 30k, 60k, 90k, 120k, 140k, and 143k. This provides 24 model-checkpoint combinations. All models are loaded via TransformerLens (Nanda & Bloom, 2022) using `HookedTransformer`, which provides clean access to intermediate activations and attention patterns.

## 3.2 Accessibility Terms and Evaluation Prompts

We select three multi-token web accessibility terms as our evaluation domain: **"screen reader,"** **"skip link,"** and **"alt text."** These terms were chosen because they are: (a) multi-token, requiring the model to bind constituent tokens into a coherent concept; (b) domain-specific, with clear factual ground truth; and (c) practically important for accessibility-aware AI systems.

For each term, we construct two types of evaluation prompts (12 total):

- **Recognition (6 prompts).** Four-choice multiple-choice questions testing factual knowledge (e.g., "A screen reader is primarily used by: A) Blind users B) Colorblind users C) Deaf users D) Mobility impaired users"). Scored via log-probability ranking: for each candidate answer, we compute the length-normalized log-probability $\frac{1}{|c|} \sum_{i} \log P(c_i \mid \text{prompt}, c_{<i})$ and select the highest-scoring choice. This follows the standard approach used by lm-eval-harness (Gao et al., 2021) for base (non-instruction-tuned) models.

- **Generation (6 prompts).** Open-ended completions testing conceptual understanding (e.g., "In web accessibility, a screen reader is"). Scored via a keyword rubric: we count word-boundary matches against a curated keyword list per term (e.g., "blind," "visual," "assistive," "software" for "screen reader"), normalize to a threshold of 3 keywords, and apply contradiction penalties. This yields a score in [0, 1].

The **behavioral score** for each checkpoint is the average across all 12 prompts: $\text{Beh} = \frac{1}{2}(\text{RecAcc} + \text{GenScore})$.

### Dataset Expansion: 9-Term and 21-Term Validation Sets

To address concerns about sample size and enable robust causal inference, we created two expanded datasets:

**9-term dataset — two variants.** We use two distinct 9-term sets depending on the analysis:

- **Set A** (`expanded_summary.csv`; used for C1/C4 Pythia lifecycle): original 3 terms plus color contrast, focus indicator, heading structure, aria attribute, **tab order**, **form validation**. This set was used to compute the coupling-decoupling Spearman correlations (§4.2).

- **Set B** (`expanded_terms_100.jsonl`; 11 prompts/term = 99 prompts; used for C3 cross-scale, C5 9-term ablation, and OLMo): original 3 terms plus color contrast, focus indicator, heading structure, aria attribute, **keyboard navigation**, **landmark region**. Set B was created for the high-density prompt evaluation and is the basis for OLMo binding/behavioral data.

The substitution of tab order and form validation (Set A) for keyboard navigation and landmark region (Set B) was made when building the 99-prompt dataset; tokenization audit (`data/tokenization/tokenization_olmo_9terms.csv`) confirms Set A terms are equally valid for OLMo. Landmark region is known to have keyword-rubric evaluation issues (§4.1.2). Cross-architecture C1/C4 comparisons involving OLMo (which uses Set B) are therefore approximate relative to the Pythia lifecycle results (which use Set A). Both sets provide 432 model-checkpoint-term observations (9 terms × 3 models × 8 checkpoints × 2 prompt types) and span diverse accessibility domains (visual, motor, cognitive, semantic).

**21-term tier-1/2/3 dataset (231 prompts).** For C5 causal ablation—which requires large N for reliable specificity estimates—we further expanded to 21 *new* terms (distinct from the 9-term set) organized in three tiers:
- **Tier 1 (AT hardware/core):** 9 foundational assistive-technology concepts (braille display, screen magnifier, voice control, switch access, audio description, captions closed, cognitive load, high contrast, keyboard shortcut)
- **Tier 2 (technical implementations):** 7 terms (keyboard navigation, focus management, skip navigation, text resize, non-text content, error identification, input purpose)
- **Tier 3 (WAI-ARIA/WCAG):** 5 terms (reflow content, text spacing, live region, alert dialog, tree grid)

This structure provides **N=105 recognition prompts** (21 terms × 5 prompts/term), reducing per-prompt noise from ±16.7 pp (N=6) to ±0.95 pp (N=105), enabling detection of small but reliable causal effects. The tier structure ensures coverage across the accessibility domain while maintaining multi-token compositionality.

*Note:* The tier-1/2/3 dataset was created as an intermediate expansion but **superseded by the canonical 41-term register** for all reported C5 experiments. The 41-term set provides N=205 prompts (vs. 105), better cross-model comparability, and term-agnostic validity. Tier-1/2/3 data were collected for Pythia lifecycle analysis but not analyzed for this paper.

**Wave-2 term set (132 prompts; collected, not analyzed for main paper).** A systematic audit of the existing ~31 terms against a 50-term candidate list (filtered by tokenization validity, domain specificity, and behavioral scorability) identified 12 recommended additions covering four categories absent from all prior sets:

| Term | Category | Tokens | Role |
|---|---|---|---|
| contrast ratio | Visual | 2 | Upgrades `color contrast` (WCAG 1.4.3 metric) |
| eye tracking | Motor-AT | 2 | New AT device class |
| time limits | Cognitive-WCAG | 2 | WCAG 2.2.1 Success Criterion |
| reduced motion | Vestibular | 2 | CSS `prefers-reduced-motion` |
| focus trap | Web-Focus | 2 | Modal dialog keyboard pattern |
| sign language | Hearing | 2 | Distinct from captions (Deaf community) |
| touch target size | Mobile | 3 | WCAG 2.5.5 (44×44 px rule) |
| haptic feedback | Sensory | 3 | Tactile output; first Sensory category term |
| plain language | Cognitive | 2 | Cognitive a11y standard |
| motion sensitivity | Vestibular | 2 | Cause paired with `reduced motion` response |
| semantic html | Web-Structure | 2 | HTML5 structural elements |
| orientation support | Mobile | 2 | WCAG 1.3.4 |

Prompts generated at `data/prompts/expanded_terms_wave2.jsonl` (132 prompts; 11 per term following the same template as tier-1/2/3). Selection methodology and full tokenization audit in `src/tokenization_audit.py`.

*Note:* Wave-2 binding and behavioral data were collected for all Pythia checkpoints (24 files each). Analysis (`src/analyze_wave2.py`) confirms the binding-behavior lifecycle pattern generalizes to these 12 additional terms (e.g., ρ(EB*, Beh) = +0.730 at 160M, +0.873 at 2.8B), but wave-2 results are not integrated into the main paper figures to maintain focus on the canonical 41-term cross-architecture comparison.

**Canonical 41-term prompt register.** For all cross-model C5 causal ablation experiments, we use a single deduplicated prompt file `data/prompts/canonical_45terms.jsonl` containing **41 unique terms** (T-V2 ∪ T-V3 ∪ T-V4) and **N=205 recognition prompts**. This register is the single source of truth for all C5 and C1-B/C4-B experiments across all models. All 41 terms were verified to tokenize as ≥2 tokens in every target model’s tokenizer (tokenization audit: `data/tokenization/tokenization_new_models_45terms.csv`).

### C1 and C4 Analysis: Two Complementary Variants

All lifecycle (C1) and decoupling (C4) analyses are reported in two complementary variants:

- **C1-A / C4-A (between-term Spearman):** Applied to the 9-term pilot set (T-V2), which was selected to maximise EB* variance across terms (a prerequisite of the between-term Spearman test). C1-A computes Spearman(ΔEB*, ΔBeh) across 9 terms at each checkpoint; C4-A splits the lifecycle into early and late windows and reports the sign change in ρ.

- **C1-B / C4-B (within-term temporal precedence):** Applied to all 41 terms. For each term *t* independently, C1-B tests whether EB*(t, k) predicts Beh(t, k+1) better than the reverse using a 1-step forward lag correlation: `r_forward(t) = Pearson(EB*(t, ck_{0:6}), Beh(t, ck_{1:7}))`. This follows the cross-lagged panel model approach for testing temporal precedence (Hamaker et al., 2015). The population-level claim (H1: EB* leads in >50% of terms) is tested with a binomial test across all terms (Wilson, 1927). C4-B computes per-term Spearman ρ in early and late windows independently and reports the fraction of terms showing strict decoupling (rho_early > 0 ∧ rho_late ≤ 0). This test requires no between-term EB* variance and generalises to any term set.

The two variants are complementary: C1-A/C4-A provides a high-contrast demonstration of the lifecycle on a carefully selected pilot set; C1-B/C4-B provides a term-agnostic validity check at scale. Results reported in §4.2 (C1-A/B) and §4.4 (C4-A/B).

### Cross-Architecture Validation: Four Additional Models

To test whether the binding-behavior lifecycle generalizes beyond the Pythia model suite, we include four additional architectures spanning a wide range of scales, training durations, and data compositions:

| Model | Parameters | Architecture | Tokenizer | Training tokens | Checkpoints |
|---|---|---|---|---|---|
| OLMo-1B | 1B | GPT-NeoX | Dolma BPE | ~59B | 8 (step 0–143k) |
| CRFM GPT-2 Small | 117M | GPT-2 | GPT-2 BPE | ~4.2B | 8 (ck-0–400k); **5 seeds** |
| SmolLM3-3B | 3B | LLaMA-3 GQA | LLaMA-3 BPE | ~11T | 8 (step 40k–3440k) |
| Qwen2.5-1.5B | 1.5B | Qwen2 GQA | Qwen tiktoken | ~18T | Final only |

**OLMo-1B** (Groeneveld et al., 2024) tests architecture generalization: trained on Dolma with a different tokenizer (Dolma BPE vs. GPT-NeoX), it starts with anomalously high EB* at step0 (0.54 vs. Pythia’s 0.15), providing a strong test of whether the lifecycle pattern is architecture-independent.

**Stanford CRFM GPT-2 Small** (`stanford-crfm/alias-gpt2-small-x21` and 4 additional seeds x49, x81, x21+, x55; collectively covering 5 random initializations) is a GPT-2-scale model trained on The Pile with intermediate checkpoints available every 100 steps up to 400k. It provides the cleanest test of the early-training dynamics at 117M scale, with 5 seeds enabling mean±SD reporting for all lifecycle and ablation statistics.

**SmolLM3-3B** (HuggingFaceTB) is a multilingual model trained on 11T tokens with a LLaMA-3 architecture and GQA. It provides a test of whether the lifecycle pattern holds in multilingual data and long-horizon training (3440k steps vs. Pythia’s 143k). Note that SmolLM3 checkpoints begin at step40k (the earliest publicly available), meaning the very early coupling phase is unobservable; C1-B results for SmolLM3 are therefore **left-censored** (the binding-precedes-behavior phase precedes our observation window, not terminates after it).

**Qwen2.5-1.5B** (Qwen Team, 2024) is a modern model trained on 18T tokens with Qwen2 GQA architecture. No intermediate checkpoints are publicly available, so lifecycle analysis (C1/C4) is structurally impossible; only single-checkpoint analyses (C3 few-shot and C5 causal ablation) are reported.

**Checkpoint alignment across models.** To enable conceptually valid cross-architecture comparisons, we map each model's selected checkpoints to approximate training token counts:

| Stage | Pythia (tokens) | OLMo (tokens) | CRFM GPT-2 (tokens) | SmolLM3 (tokens) |
|-------|----------------|---------------|---------------------|-----------------|
| Init | 0 | 0 | 0 | — |
| Early | ~31B (step 15k) | ~7B (step 15k) | ~209M (ck-1000) | ~375B (step 40k) |
| Mid-1 | ~125B (step 60k) | ~30B (step 60k) | ~838M (ck-4000) | ~1.9T (step 200k) |
| Mid-2 | ~188B (step 90k) | ~45B (step 90k) | ~2.1B (ck-10000) | ~3.8T (step 400k) |
| Late | ~300B (step 143k) | ~60B (step 143k) | ~4.2B (ck-400000) | ~32T (step 3440k) |

CRFM training duration (~4.2B tokens) is substantially shorter than Pythia's (~300B), OLMo's (~60B), and SmolLM3's (~32T). This means CRFM lifecycle comparisons should be interpreted as "same number of training steps at ~10× lower data" rather than equivalent training maturity. Pythia and OLMo are most comparable in token-aligned terms.

**Decoding protocol.** We use greedy decoding (temperature = 0) for generation tasks to maximize determinism and reproducibility. Variability experiments (§4.1.3) confirm greedy decoding shows lowest variability at trained checkpoints (std = 0.124 at 1B step 143k vs. 0.294-0.328 for sampling temperatures), validating this choice for main experiments. Recognition scoring is deterministic (log-probability ranking) and unaffected by decoding parameters.

## 3.3 Attention-Head Binding Metrics

### Attention Convention

We write $A_{l,h}[i,j]$ for the attention weight in layer $l$, head $h$ from query position $i$ to key position $j$. Thus $A_{l,h}[i,j]$ with $i > j$ represents a later token attending to an earlier token (later-to-earlier attention flow).

### Binding Strength Index (BSI)

For a term span occupying token positions $\{s_1, s_2, \ldots, s_k\}$, the BSI at layer $l$, head $h$ measures the average later-to-earlier attention within the span (Clark et al., 2019; Haviv et al., 2023; Miletić & Schulte im Walde, 2024):

$$\text{BSI}_{l,h} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} A_{l,h}[s_i, s_j]$$

where $\mathcal{P} = \{(i,j) : s_i > s_j\}$ is the set of later-to-earlier token pairs within the term span.

**Metric justification.** We focus on later-to-earlier intra-span attention (BSI) rather than alternatives such as attention entropy or bidirectional attention flow for theoretical and empirical specificity. BSI directly measures the hypothesized binding pattern—later tokens attending back to earlier tokens within a concept span—capturing the directional dependency structure we expect for compositional binding (e.g., "reader" attending to "screen" to form "screen reader"). Attention entropy measures diffuseness but not directionality, while general attention flow does not isolate intra-concept binding from contextual dependencies. Our analysis (§4.1.2) shows BSI and entropy are complementary: low entropy (focused attention) is necessary but not sufficient for high BSI, confirming BSI captures a specific compositional pattern beyond general attention concentration.

While the concept of inspecting intra-span attention patterns has precedents in multi-word expression analysis, the specific directed formulation and its application to tracking concept emergence are novel to this work.

### Excess Binding (EB)

The **Excess Binding** at layer $l$ captures how much the best head exceeds the layer average:

$$\text{EB}_l = \max_h \text{BSI}_{l,h} - \frac{1}{H} \sum_{h=1}^{H} \text{BSI}_{l,h}$$

where $H$ is the number of attention heads in the layer. This measures whether binding is concentrated in specific heads (high EB) or distributed uniformly (low EB). High EB indicates specialized binding structure.

### EB\* (Aggregate Binding)

The aggregate binding metric is the maximum EB across layers:

$$\text{EB}^* = \max_l \text{EB}_l$$

EB\* serves as the primary binding metric throughout this paper. For each checkpoint, we report the mean EB\* across all 12 prompts.

### Term Span Identification

Term tokens are located in the input via exact subsequence matching of the BPE token IDs, with fallback to character-level search for aliased forms (e.g., "alternative text" for "alt text"). Multiple encoding variants are tried (bare, space-prefixed, capitalized, title-cased) to handle BPE tokenization variability.

### Memory-Efficient Extraction

Attention patterns are extracted layer-by-layer using TransformerLens's `run_with_cache` with `stop_at_layer` to limit computation and memory usage. This enables extraction on consumer GPUs even for the 2.8B model.

## 3.4 Head Ablation for Causal Testing (C5)

To test whether high-binding heads are causally necessary for task performance, we perform targeted zero-ablation: during the forward pass, the attention pattern tensor $A_{l,h}$ is set to zero for selected heads via TransformerLens forward hooks.

For each model, we identify the top-$k$ heads by average BSI across all prompts, then evaluate under four conditions:

1. **Baseline:** No ablation.
2. **Top-$k$ ablation:** Zero the $k$ highest-BSI heads.
3. **Random ablation:** Zero $k$ randomly selected heads (5 trials, averaged).
4. **Bottom-$k$ ablation:** Zero the $k$ lowest-BSI heads (negative control).

We use $k = 4$ for all models. The **specificity** of the causal effect is measured as the difference between the top-$k$ accuracy drop and the mean random accuracy drop: $\text{Spec} = \Delta\text{Acc}_{\text{top}} - \bar{\Delta}\text{Acc}_{\text{rand}}$. Positive specificity indicates binding heads are selectively necessary; negative specificity indicates a suppressor pattern.

**Canonical 41-term ablation protocol.** All cross-architecture C5 results use a single deduplicated prompt file `data/prompts/canonical_45terms.jsonl` (N=205 recognition prompts across 41 terms). Recognition accuracy is the primary metric for cross-model comparability (generation scoring varies by model output length conventions). We apply this protocol to all seven models: Pythia-160M, Pythia-1B, Pythia-2.8B, OLMo-1B, CRFM GPT-2 Small (5 seeds), SmolLM3-3B, and Qwen2.5-1.5B. Implementation: `src/run_c5_canonical.py` for Pythia; `src/run_c5_new_models.py` for cross-architecture models.

## 3.5 Few-Shot Unlockability Testing (C3)

To test whether binding structure contains latent knowledge that prompting can unlock, we compare zero-shot and few-shot generation performance. We prepend a single worked example (one-shot) as the exemplar for the evaluation prompt. The improvement from zero-shot to few-shot (in percentage points) is the **unlockability score** Δ.

**9-term unified protocol.** For Pythia 3×2 (three scales × early/late) and cross-architecture OLMo and CRFM runs, we use the standardised `eval_few_shot_c3.py` script with term-specific multi-sentence exemplars (`data/prompts/expanded_terms_100.jsonl`, Set B). This yields N=54 generation prompts per model-checkpoint pair (9 terms × 6 prompts/term). Exemplars are constructed to be semantically complete (not simple definition repetitions) to minimize in-context copying.

**41-term canonical protocol.** For all cross-architecture models (OLMo, CRFM, SmolLM3, Qwen) and the full Pythia scale sweep, we apply `eval_few_shot_c3_expanded.py` using the canonical prompt register (`data/prompts/canonical_45terms.jsonl`). This yields N=246 generation prompts per model-checkpoint pair (41 terms × 6 prompts/term). This protocol is the primary source for all cross-architecture Δ comparisons reported in §4.3.

**Headroom compression.** Models with high zero-shot baselines (ZS ≳ 0.50) show lower nominal Δ even when genuine coupling is present, because the few-shot ceiling is shared across models. We interpret such cases (SmolLM3 ZS≈0.49–0.51, Qwen ZS≈0.54) as **headroom-compressed** rather than weakly coupled: the same absolute few-shot score (~0.70–0.72) is achieved, but the starting floor is higher.

**Paraphrase-exemplar control.** To distinguish genuine knowledge retrieval from in-context phrasing copying, we run a paraphrase condition at step15k for all three Pythia scales (`src/eval_c3_paraphrase.py`): the exemplar is rewritten to use different surface phrasing while preserving factual content. The gap between FS-original and FS-paraphrase quantifies the copying contribution; the gap between FS-paraphrase and zero-shot quantifies genuine knowledge retrieval. Results reported in §4.3.

## 3.6 Implementation Details

All experiments are conducted on a single NVIDIA GPU (15GB VRAM). Models are loaded in float32 precision via TransformerLens. Behavioral evaluation uses greedy decoding (temperature = 0) for generation tasks unless otherwise specified. Random baselines for ablation use a fixed seed (42) for reproducibility. Code is available at [repository URL].

**Sampling parameter robustness.** To assess sensitivity to decoding stochasticity, we conducted variability experiments on 6 representative checkpoints (160M, 1B, 2.8B × early/late training) using:
- **Temperatures:** 0.0 (greedy), 0.3 (low sampling), 0.7 (moderate sampling)
- **Random seeds:** 5 independent replicates per temperature (seeds: 42, 123, 456, 789, 1024)
- **Total runs per checkpoint:** 15 (3 temperatures × 5 seeds)

This design enables measurement of both temperature-induced variability and seed-induced stochasticity. Recognition scoring is deterministic (log-probability ranking) and unaffected by these parameters, so variability analysis focuses on generation tasks. Results are reported in §4.1.3.
