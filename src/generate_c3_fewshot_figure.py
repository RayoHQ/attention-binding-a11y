"""Generate C3 few-shot unlockability figure.

Two-panel figure:
  Panel A — Grouped bar chart: Pythia 3×2 (early/late) showing Δ (pp) with zero-shot
            and few-shot means annotated.  Data: 9-term unified protocol (N=54 gen prompts).
  Panel B — Cross-model Δ comparison at trained checkpoints. Data: 41-term canonical
            protocol (N=246 gen prompts, all models). Pythia also shown at early ck.

All Panel A data from results.md §4.3 unified protocol (9 terms).
All Panel B data from data/results/few_shot_c3_expanded/ (41 terms).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


# --------------------------------------------------------------------------- #
# Data — Pythia unified protocol (9 terms, N=54 gen prompts)
# --------------------------------------------------------------------------- #
pythia_data = {
    "160M": {"early": (0.265, 0.630), "late": (0.290, 0.599)},
    "1B":   {"early": (0.340, 0.704), "late": (0.395, 0.667)},
    "2.8B": {"early": (0.422, 0.710), "late": (0.506, 0.700)},
}

# --------------------------------------------------------------------------- #
# Data — Cross-model (41-term canonical protocol, N=246 gen prompts)
# Source: data/results/few_shot_c3_expanded/*.json
# Format: (label, zs, fs, checkpoint_label, family)
# --------------------------------------------------------------------------- #
cross_model = [
    ("Pythia-160M\nearly",  0.328, 0.653, "step15k",   "pythia"),
    ("Pythia-160M\nlate",   0.321, 0.648, "step143k",  "pythia"),
    ("Pythia-1B\nearly",    0.362, 0.713, "step15k",   "pythia"),
    ("Pythia-1B\nlate",     0.365, 0.734, "step143k",  "pythia"),
    ("Pythia-2.8B\nearly",  0.467, 0.791, "step15k",   "pythia"),
    ("Pythia-2.8B\nlate",   0.503, 0.740, "step143k",  "pythia"),
    ("OLMo-1B\nearly",      0.416, 0.697, "step15k",   "olmo"),
    ("OLMo-1B\nlate",       0.478, 0.694, "step143k",  "olmo"),
    ("CRFM x̄\nlate",       0.072, 0.147, "ck-400k",   "crfm"),
    ("SmolLM3\nlate",       0.508, 0.698, "step3440k", "smollm3"),
    ("Qwen2.5\nfinal",      0.542, 0.724, "final",     "qwen"),
]

FAMILY_COLORS = {
    "pythia":  "#3b82f6",
    "olmo":    "#22c55e",
    "crfm":    "#f59e0b",
    "qwen":    "#8b5cf6",
    "smollm3": "#ec4899",
}

EARLY_COLOR = "#1d4ed8"
LATE_COLOR  = "#93c5fd"

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- Panel A: Pythia stacked-style grouped bar (ZS + Δ on top) ----
sizes = ["160M", "1B", "2.8B"]
x = np.arange(len(sizes))
width = 0.30

for offset, phase, color_zs, color_delta, label in [
    (-width/2 - 0.02, "early", "#bfdbfe", EARLY_COLOR, "Early (step15k)"),
    (+width/2 + 0.02, "late",  "#dbeafe", LATE_COLOR,  "Late (step143k)"),
]:
    for i, size in enumerate(sizes):
        zs, fs = pythia_data[size][phase]
        delta  = fs - zs
        xi = x[i] + offset
        # ZS base bar
        ax1.bar(xi, zs,    width, color=color_zs,    edgecolor='black', linewidth=0.7, alpha=0.9)
        # Delta on top
        ax1.bar(xi, delta, width, bottom=zs,
                color=color_delta, edgecolor='black', linewidth=0.7, alpha=0.9)
        # Delta annotation
        ax1.text(xi, fs + 0.012, f"+{delta*100:.0f}pp",
                 ha='center', va='bottom', fontsize=8,
                 color=EARLY_COLOR if phase == "early" else LATE_COLOR, fontweight='bold')

ax1.axhline(0, color='black', linewidth=0.8)

# Threshold line at 20pp (0.20 Δ minimum support)
ax1.axhline(0.0, color='black', linewidth=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(["Pythia-160M", "Pythia-1B", "Pythia-2.8B"], fontsize=9.5)
ax1.set_ylabel("Generation score (ZS = base, Δ = few-shot gain)", fontweight='bold')
ax1.set_title("A. Pythia C3 Unlockability (9-term unified protocol)\n"
              "N=54 generation prompts, early vs. late checkpoint",
              fontweight='bold', pad=10)
ax1.set_ylim(0, 0.85)
ax1.grid(axis='y', alpha=0.25, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

legend_a = [
    mpatches.Patch(color="#bfdbfe", edgecolor='black', label='Zero-shot (early ck)'),
    mpatches.Patch(color=EARLY_COLOR, edgecolor='black', label='Few-shot gain Δ (early)'),
    mpatches.Patch(color="#dbeafe", edgecolor='black', label='Zero-shot (late ck)'),
    mpatches.Patch(color=LATE_COLOR, edgecolor='black', label='Few-shot gain Δ (late)'),
]
ax1.legend(handles=legend_a, loc='upper left', framealpha=0.9, fontsize=8)

# ---- Panel B: Cross-model Δ comparison ----
deltas     = [(fs - zs) * 100 for _, zs, fs, *_ in cross_model]
labels     = [m[0] for m in cross_model]
families   = [m[4] for m in cross_model]
bar_colors = [FAMILY_COLORS[f] for f in families]
checkpoints= [m[3] for m in cross_model]

x2 = np.arange(len(cross_model))
bars2 = ax2.bar(x2, deltas, color=bar_colors, edgecolor='black',
                linewidth=0.7, alpha=0.85, width=0.6, zorder=3)

# Reference lines
ax2.axhline(0,  color='black',   linewidth=1.2, linestyle='-',  zorder=2)
ax2.axhline(20, color='#0369a1', linewidth=1.2, linestyle='--', alpha=0.7,
            label='20pp support threshold', zorder=2)

# Value labels
for bar, d in zip(bars2, deltas):
    ypos = d + (0.8 if d >= 0 else -2.5)
    va   = 'bottom' if d >= 0 else 'top'
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{d:+.1f}", ha='center', va=va, fontsize=7.5, fontweight='bold')

ax2.set_xticks(x2)
ax2.set_xticklabels(labels, fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
ax2.set_ylabel("Few-shot Δ (percentage points)", fontweight='bold')
ax2.set_title("B. Cross-Model C3 Few-Shot Δ\n"
              "All models, 41-term canonical protocol (N=246 gen prompts)",
              fontweight='bold', pad=10)
ax2.set_ylim(0, 43)
ax2.grid(axis='y', alpha=0.25, linestyle='--', zorder=1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

legend_b = [
    mpatches.Patch(color=FAMILY_COLORS['pythia'],  label='Pythia'),
    mpatches.Patch(color=FAMILY_COLORS['olmo'],    label='OLMo-1B'),
    mpatches.Patch(color=FAMILY_COLORS['crfm'],    label='CRFM GPT-2 Sm'),
    mpatches.Patch(color=FAMILY_COLORS['qwen'],    label='Qwen2.5-1.5B'),
    mpatches.Patch(color=FAMILY_COLORS['smollm3'], label='SmolLM3-3B'),
    plt.Line2D([0], [0], color='#0369a1', linewidth=1.5, linestyle='--',
               label='20pp threshold'),
]
ax2.legend(handles=legend_b, loc='upper right', framealpha=0.9, fontsize=8)

# Landmark caveat footnote
fig.text(0.02, -0.03,
         "Panel B uses 41-term canonical protocol (N=246) for all models. "
         "Panel A uses 9-term unified protocol (N=54) for Pythia only. "
         "CRFM shows mean across seeds 1–5 at ck-400k.",
         fontsize=7.5, color='#555', style='italic')

plt.tight_layout()

output_dir = Path('paper/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'c3_fewshot_unlockability.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'c3_fewshot_unlockability.pdf', bbox_inches='tight')

print("✅ Generated C3 few-shot unlockability figure")
print(f"   paper/figures/c3_fewshot_unlockability.png")
print(f"   paper/figures/c3_fewshot_unlockability.pdf")
