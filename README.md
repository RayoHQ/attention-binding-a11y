# Attention-Head Binding as a Mechanistic Marker of Accessibility Concept Emergence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and data for the paper *"Attention-Head Binding as a Term-Conditioned Mechanistic Marker of Accessibility Concept Emergence in Language Models"* (Tran, 2026).

## Overview

In this TMLR paper, we introduce **EB\*** (effective binding), a mechanistic interpretability metric that tracks how attention heads bind multi-token accessibility terms (e.g., "screen reader," "alt text") during training. Using **seven models** across **five architectures** (Pythia 160M/1B/2.8B, OLMo-1B, CRFM GPT-2 Small, SmolLM3-3B, Qwen2.5-1.5B) and **41 accessibility terms** (N=205 prompts), we demonstrate:

- **Discriminant Validity (V2–V4):** EB\* validates against token co-occurrence baselines, establishing a clear gradient from nonsense (0.26) to real terms (0.74), all p < 0.001, Cohen's d = 1.2–2.9
- **C1 (Lead-lag emergence):** Binding precedes behavioral competence with phase transition: early coupling (ρ = +0.57, p < 0.001) reverses to decoupling (ρ = −0.20, p = 0.01) at trained checkpoints; replicated across OLMo-1B (90% EB\*-leads) and CRFM (72.7%)
- **C3 (Unlockability):** Few-shot prompting yields gains up to +61 pp (183% relative) when EB\* > 0.6; Pythia-1B shows strongest cross-architecture effect (+37.0 pp); modern models (SmolLM3, Qwen) exhibit headroom compression (+18–19 pp)
- **C4 (Decoupling):** Two-factor model emerges — parameter threshold (~1B) governs decoupling depth; training-step threshold (~300K) governs temporal ordering
- **C5 (Causal regimes):** Cross-scale reversal confirmed — binding heads necessary at 160M (−16.7 pp), functionally superseded at 2.8B (+33.3 pp); OLMo/Qwen show ceiling effects, SmolLM3 distributed regime, CRFM initialization sensitivity (4/5 seeds coupled, 1/5 suppressor)

## Repository Structure

```
attention-binding-a11y/
├── src/                            # Source code
│   ├── utils_model.py              # Model loading with checkpoint support
│   ├── scoring.py                  # Recognition and generation scoring
│   ├── eval_behavior.py            # Behavioral probe evaluation
│   ├── extract_attention.py        # Attention extraction, BSI/EB/EB* metrics
│   ├── tokenization_audit.py       # Tokenization span verification
│   ├── analysis_pilot.py           # Correlation and Go/No-Go analysis
│   ├── minimal_causal.py           # C5: 160M head ablation
│   ├── minimal_causal_28b.py       # C5: 2.8B head ablation
│   └── eval_few_shot.py            # C3: Few-shot unlockability testing
├── data/
│   ├── prompts/
│   │   ├── pilot_terms.jsonl       # 12 prompts (3 terms × 2 tasks × 2 variants)
│   │   ├── expanded_99_prompts/  # 99 prompts (9 terms × 11 formats) for robustness
│   │   └── canonical_41_terms/     # 41 accessibility terms (N=205 prompts)
│   ├── results/
│   │   ├── behavioral/             # Behavioral probe scores
│   │   ├── binding/                # EB* binding metrics
│   │   ├── causal/                 # C5 ablation results
│   │   └── few_shot/               # C3 unlockability results
│   └── tokenization/               # Tokenization tables
├── config/
│   └── pilot.yaml                  # Experiment configuration
├── notebooks/
│   ├── figure1_emergence_curves.ipynb  # Figures 1 & 4
│   ├── verify_checkpoints_v2.ipynb     # Checkpoint verification
│   └── verify_setup.ipynb              # Environment check
├── figures/                        # Generated figures
├── paper/                          # Paper source (Markdown)
│   ├── main.md
│   ├── sections/
│   └── appendix/
├── tests/
│   └── test_behavioral.py          # Unit tests
├── requirements.txt
├── setup_data.py                   # Environment setup script
├── REPRODUCTION_CHECKLIST.md
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU with compute capability 7.0+
- **VRAM requirements by model:**
  - 8GB: Pythia-160M, CRFM GPT-2 Small (117M)
  - 12GB: Pythia-1B, OLMo-1B, Qwen2.5-1.5B
  - 16GB+: Pythia-2.8B, SmolLM3-3B (batch size may need reduction)

### Setup

```bash
git clone https://github.com/RayoHQ/attention-binding-a11y.git
cd attention-binding-a11y

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Verify environment
python setup_data.py
```

Pythia model checkpoints are downloaded automatically from HuggingFace when running experiments.

## Quick Start

### Reproduce All Main Results

```bash
# 1. Verify tokenization spans
python src/tokenization_audit.py

# 2. Extract binding metrics (repeat for each model/checkpoint)
python src/extract_attention.py 160m step120000

# 3. Run behavioral evaluation
python src/eval_behavior.py 160m step120000

# 4. C3: Few-shot unlockability
python src/eval_few_shot.py

# 5. C5: Causal ablation
python src/minimal_causal.py        # 160M
python src/minimal_causal_28b.py    # 2.8B

# 6. Summary statistics and correlations
python src/analysis_pilot.py 160m
```

### Expected Key Outputs

| Experiment | Output | Key Metric |
|-----------|--------|------------|
| Discriminant Validity | `data/results/controls/v2_*`, `v3_*`, `v4_*` | d = 1.2–2.9, p < 0.001 |
| C1 (Lead-lag) | `data/results/binding/*_binding.jsonl` | Spearman ρ = +0.57 early, −0.20 late |
| C3 (Unlockability) | `data/results/few_shot/*_few_shot.json` | +61.1 pp at 160M step15k (pilot); +37.0 pp Pythia-1B; +18–19 pp modern models |
| C4 (Decoupling) | `data/results/pilot_summary.csv` | 1B EB\* plateau |
| C5 (Causal) | `data/results/causal/*_causal.json` | 160M: −16.7 pp, 2.8B: +33.3 pp; cross-architecture patterns |

### Approximate Runtime

| Task | GPU | CPU |
|------|-----|-----|
| Tokenization audit | 5 min | 10 min |
| Single checkpoint (binding + behavior, 205 prompts) | 2–12 min | 5–20 min |
| Full pilot (24 checkpoints, 9 terms, for initial validation) | 2–4 hours | 6–12 hours |
| C3 unlockability (few-shot across 7 models) | 2–4 hours | 4–8 hours |
| C5 ablation (causal head ablation, 5 models) | 6–10 hours | 4–8 hours |
| Discriminant validity controls (V1–V4) | 30 min | 1 hour |
| Prompt robustness (99 prompts) | 30 min | 1 hour |
| 41-term cross-architecture (81 checkpoints, 7 models) | 4–6 hours | 10–16 hours |
| **Reproducible final results** | **~20–25 hours GPU** | **~50–70 hours CPU** |
| *(Total project effort, incl. Feb pilot, April expansion & debugging)* | *~40–60 hours GPU* | *~100–140 hours CPU* |

**Storage:** ~50–70GB for model checkpoints (Pythia suite auto-downloads from HuggingFace; OLMo, CRFM, SmolLM3, and Qwen add ~20–30GB)

## Key Results

| Claim | Finding | Section |
|-------|---------|---------|
| **Discriminant Validity** | Gradient: nonsense (0.26) → rare pairs (0.50) → real terms (0.74), p < 0.001 | §4.1 |
| **C1** | Phase transition: early coupling (ρ = +0.57) → late decoupling (ρ = −0.20) | §4.3 |
| **C3** | +61 pp few-shot improvement (pilot); +37.0 pp Pythia-1B strongest; +18–19 pp modern models with headroom compression | §4.5 |
| **C4** | 1B binding saturates at step 15k; behavior improves through step 143k | §4.4 |
| **C5** | 160M: ablation impairs (−16.7 pp); 2.8B: ablation helps (+33.3 pp); OLMo/Qwen ceiling; CRFM initialization sensitivity | §4.6 |

## Citation

```bibtex
@article{tran2026binding,
  title={Attention-Head Binding as a Term-Conditioned Mechanistic Marker of Accessibility Concept Emergence in Language Models},
  author={Tran, Khanh-Dung},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=QG7mfCy9mu}
}
```

## Paper Compilation

The paper source is in `paper/` as Markdown. To compile to PDF:

```bash
# Install pandoc
sudo apt-get install pandoc texlive-latex-base texlive-latex-extra

# Compile all sections into a single PDF
cd paper
pandoc main.md sections/introduction.md sections/related_work.md \
       sections/methods.md sections/results.md sections/discussion.md \
       sections/conclusion.md appendix/raw_data.md \
       -o attention_binding_a11y.pdf \
       --pdf-engine=pdflatex \
       -V geometry:margin=1in
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

I am deeply grateful to Professor Manolis Kellis, the Mantis team, and my classmates from the Generative AI course (January 5, 2026) for many stimulating intellectual exchanges. Through this course, I gained a strong conceptual grounding in research ethics and the personal confidence to pursue this work.

I also thank the TMLR reviewers and action editors for their voluntary and rigorous engagement. Their feedback was instrumental in expanding this work from a focused Pythia mechanistic interpretability study into a cross-architecture analysis spanning seven models across five architectures — with discriminant validity controls, causal ablations, and few-shot unlockability experiments that substantially strengthened the empirical grounding. Any errors of interpretation are my own.

This work also builds directly on and extends prior behavioral analysis of accessibility knowledge in Pythia models by Trisha Salas (Salas, 2026), whose exploratory work on February 1, 2026 motivated the choice of accessibility concepts as the case study for this paper. That work established that accessibility concepts such as *"screen reader"* and *"alt text"* emerge behaviorally at different rates across model scales. The present study extends this line of inquiry by shifting from **behavioral evaluation** to **mechanistic analysis**, introducing **EB\*** as an attention-based binding metric to probe *how* and *when* these concepts emerge internally during training, and how their causal role changes with scale.
