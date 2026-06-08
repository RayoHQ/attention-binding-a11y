"""Generate C5 cross-architecture causal specificity figure.

Two-panel figure:
  Panel A — Bar chart of rec-only specificity for all 7 models (canonical41 N=205).
  Panel B — Top-ablation Δ vs. random-ablation Δ scatter, one point per model/seed.

All data from results.md §4.5.5 table.
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
# Data — Panel A (rec-only specificity = top_drop − rand_drop)
# --------------------------------------------------------------------------- #
panel_a = [
    {"label": "Pythia-1B",          "spec": +0.117, "err": None,  "regime": "coupled"},
    {"label": "Pythia-160M",        "spec": +0.137, "err": None,  "regime": "coupled"},
    {"label": "Pythia-2.8B",        "spec": +0.110, "err": None,  "regime": "redundancy"},
    {"label": "CRFM mean\n(5-seed)","spec": +0.081, "err": 0.152, "regime": "coupled"},
    {"label": "Qwen2.5-1.5B",       "spec": +0.005, "err": None,  "regime": "ceiling"},
    {"label": "OLMo-1B",            "spec": -0.006, "err": None,  "regime": "ceiling"},
    {"label": "SmolLM3-3B",         "spec": -0.043, "err": None,  "regime": "ceiling"},
]

REGIME_COLORS = {
    "coupled":    "#3b82f6",
    "redundancy": "#06b6d4",
    "ceiling":    "#9ca3af",
}

REGIME_LABELS = {
    "coupled":    "Coupled (load-bearing)",
    "redundancy": "Redundancy regime",
    "ceiling":    "Ceiling / distributed",
}

# --------------------------------------------------------------------------- #
# Data — Panel B (top Δ vs rand Δ, one point per model/seed)
# All values in pp (percentage points).
# --------------------------------------------------------------------------- #
panel_b = [
    # (label,          top_delta,  rand_delta,  regime,       marker)
    ("Pythia-1B",       -15.1,      -3.4,        "coupled",    "o"),
    ("Pythia-160M",     -11.2,      +2.4,        "coupled",    "o"),
    ("Pythia-2.8B",      -7.3,      +3.7,        "redundancy", "s"),
    ("CRFM x1",         +20.9,      +3.5,        "suppressor", "^"),
    ("CRFM x2",         -23.4,      -3.1,        "coupled",    "^"),
    ("CRFM x3",         -29.8,     -11.4,        "coupled",    "^"),
    ("CRFM x4",          -6.3,      +0.8,        "coupled",    "^"),
    ("CRFM x5",         -14.6,      -2.4,        "coupled",    "^"),
    ("OLMo-1B",          -1.0,      -1.6,        "ceiling",    "D"),
    ("SmolLM3-3B",       +3.4,      -0.9,        "ceiling",    "D"),
    ("Qwen2.5-1.5B",     -1.0,      -0.5,        "ceiling",    "D"),
]

SUPPRESSOR_COLOR = "#ef4444"
REGIME_COLORS_B = dict(REGIME_COLORS)
REGIME_COLORS_B["suppressor"] = SUPPRESSOR_COLOR

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- Panel A ----
x = np.arange(len(panel_a))
bar_colors = [REGIME_COLORS[r['regime']] for r in panel_a]
bars = ax1.bar(x, [r['spec'] for r in panel_a], color=bar_colors, edgecolor='black',
               linewidth=0.8, width=0.55, alpha=0.85, zorder=3)

# Error bar for CRFM mean
for i, r in enumerate(panel_a):
    if r['err'] is not None:
        ax1.errorbar(i, r['spec'], yerr=r['err'], fmt='none', color='black',
                     capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

# Reference lines
ax1.axhline(0,    color='black',  linewidth=1.2, linestyle='-',  zorder=2)
ax1.axhline(0.10, color='#0369a1', linewidth=1.2, linestyle='--', alpha=0.7,
            label='Support threshold (0.10)', zorder=2)

# Value labels on bars
for bar, r in zip(bars, panel_a):
    spec = r['spec']
    ypos = spec + (0.01 if spec >= 0 else -0.025)
    va   = 'bottom' if spec >= 0 else 'top'
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{spec:+.3f}", ha='center', va=va, fontsize=8.5, fontweight='bold',
             color='white' if abs(spec) > 0.03 else '#333')

ax1.set_xticks(x)
ax1.set_xticklabels([r['label'] for r in panel_a], fontsize=9, rotation=45, ha='right')
ax1.set_ylabel("Rec-only specificity\n(top_drop − rand_drop)", fontweight='bold')
ax1.set_title("A. Causal Specificity Across Models (C5)\n"
              "canonical41, N=205 prompts, k=4 heads ablated", fontweight='bold', pad=10)
ax1.set_ylim(-0.25, 0.35)
ax1.grid(axis='y', alpha=0.25, linestyle='--', zorder=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

legend_handles_a = [
    mpatches.Patch(color=REGIME_COLORS['coupled'],    label='Coupled (load-bearing)'),
    mpatches.Patch(color=REGIME_COLORS['redundancy'], label='Redundancy regime'),
    mpatches.Patch(color=REGIME_COLORS['ceiling'],    label='Ceiling / distributed'),
    plt.Line2D([0], [0], color='#0369a1', linewidth=1.5, linestyle='--',
               label='Support threshold (0.10)'),
]
ax1.legend(handles=legend_handles_a, loc='upper right', framealpha=0.9, fontsize=8)

# ---- Panel B: scatter top Δ vs rand Δ ----
diag_range = np.linspace(-35, 25, 100)
ax2.plot(diag_range, diag_range, color='#9ca3af', linewidth=1.0,
         linestyle='--', alpha=0.6, label='Top = Random (spec=0)')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.4)
ax2.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.4)

for label, top_d, rand_d, regime, marker in panel_b:
    color = REGIME_COLORS_B[regime]
    ax2.scatter(rand_d, top_d, color=color, marker=marker, s=80,
                edgecolors='black', linewidth=0.8, alpha=0.85, zorder=4)
    # Label offset to avoid overlap
    dx, dy = 0.7, 0.7
    if label == "CRFM x1":
        dx, dy = 0.7, -2.5
    elif label == "CRFM x3":
        dx, dy = -8, -2.0
    elif label == "Pythia-1B":
        dx, dy = 0.7, -2.5
    ax2.annotate(label, (rand_d, top_d), xytext=(rand_d + dx, top_d + dy),
                 fontsize=7.5, color=color,
                 arrowprops=dict(arrowstyle='-', color=color, lw=0.5, alpha=0.5))

# Quadrant shading
ax2.axhspan(-40, 0, xmin=0, xmax=1, alpha=0.04, color='blue')   # top hurts
ax2.axhspan(0, 30, xmin=0, xmax=1, alpha=0.04, color='red')      # top helps

ax2.set_xlabel("Random ablation Δ rec (pp)", fontweight='bold')
ax2.set_ylabel("Top-binding ablation Δ rec (pp)", fontweight='bold')
ax2.set_title("B. Top vs. Random Ablation Effect\n(below diagonal = top more harmful = coupled)",
              fontweight='bold', pad=10)
ax2.set_xlim(-15, 8)
ax2.set_ylim(-35, 28)
ax2.grid(alpha=0.2, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

legend_handles_b = [
    mpatches.Patch(color=REGIME_COLORS_B['coupled'],    label='Coupled'),
    mpatches.Patch(color=REGIME_COLORS_B['redundancy'], label='Redundancy'),
    mpatches.Patch(color=REGIME_COLORS_B['ceiling'],    label='Ceiling'),
    mpatches.Patch(color=SUPPRESSOR_COLOR,              label='Suppressor (CRFM x1)'),
    plt.Line2D([0], [0], color='#9ca3af', linewidth=1.2, linestyle='--',
               label='Top = Random'),
]
ax2.legend(handles=legend_handles_b, loc='upper left', framealpha=0.9, fontsize=8)

plt.tight_layout()

output_dir = Path('paper/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'c5_crossarch_specificity.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'c5_crossarch_specificity.pdf', bbox_inches='tight')

print("✅ Generated C5 cross-architecture specificity figure")
print(f"   paper/figures/c5_crossarch_specificity.png")
print(f"   paper/figures/c5_crossarch_specificity.pdf")
