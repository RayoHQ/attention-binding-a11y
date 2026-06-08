"""Generate C1-B forest plot: EB*-leads fraction with 95% Wilson CI per model.

All data from results.md §4.2.1 (41-term canonical dataset).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------- #
# Shared plot style
# --------------------------------------------------------------------------- #
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


def wilson_ci(k, n, z=1.96):
    """Return (centre, lo, hi) Wilson 95% CI for proportion k/n."""
    if n == 0:
        return 0.5, 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- #
# Data — (label, k, n, sig_star, model_family, censored)
# --------------------------------------------------------------------------- #
models = [
    # (display_label,          k,   n,   sig,     family,    censored)
    ("OLMo-1B (41t)",         36,  40, "***",   "olmo",     False),
    ("Pythia-2.8B (41t)",     27,  34, "***",   "pythia",   False),
    ("CRFM x3 (41t)",         36,  41, "***",   "crfm",     False),
    ("CRFM x2 (41t)",         32,  41, "***",   "crfm",     False),
    ("CRFM mean (5-seed)",   149, 205, "***",   "crfm",     False),
    ("Pythia-1B (41t)",       30,  41, "**",    "pythia",   False),
    ("CRFM x4 (41t)",         29,  41, "**",    "crfm",     False),
    ("CRFM x5 (41t)",         26,  41, "ns",    "crfm",     False),
    ("CRFM x1 (41t)",         26,  41, "ns",    "crfm",     False),
    ("SmolLM3-3B (41t) ‡",   21,  41, "ns",    "smollm3",  True),
    ("Pythia-160M (41t)",      3,  41, "—",     "pythia",   False),
]

FAMILY_COLORS = {
    "pythia":  "#3b82f6",   # blue
    "olmo":    "#22c55e",   # green
    "crfm":    "#f59e0b",   # amber
    "smollm3": "#a855f7",   # purple
}

# --------------------------------------------------------------------------- #
# Build figure
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(11, 6.5))

y_positions = np.arange(len(models))

for i, (label, k, n, sig, family, censored) in enumerate(models):
    y = len(models) - 1 - i        # top-to-bottom
    p, lo, hi = wilson_ci(k, n)
    color = FAMILY_COLORS[family]
    alpha = 0.45 if censored else 0.9

    # Horizontal CI line
    ax.hlines(y, lo, hi, color=color, linewidth=2.5, alpha=alpha,
              linestyle='--' if censored else '-')
    # Point estimate
    marker = 'D' if censored else 'o'
    ax.plot(p, y, marker=marker, color=color, markersize=9 if label.startswith("CRFM mean") else 7,
            alpha=alpha, markeredgecolor='white', markeredgewidth=1.2,
            zorder=5)

    # Significance star (omit "—" — it is shown via the value label position instead)
    offset = hi + 0.025
    if sig != "—":
        ax.text(min(offset, 0.97), y, sig, va='center', ha='left',
                fontsize=8.5, color=color if sig not in ("ns",) else '#888',
                fontweight='bold' if sig not in ("ns",) else 'normal')

    # Value label — left of CI normally; right of CI when lo is too small
    pct_label = f"{100*k/n:.0f}%  ({k}/{n})"
    if lo < 0.1:
        ax.text(hi + 0.03, y, pct_label, va='center', ha='left',
                fontsize=8, color='#555')
    else:
        ax.text(lo - 0.015, y, pct_label, va='center', ha='right', fontsize=8,
                color='#555')

# Reference lines
ax.axvline(0.5,  color='#dc2626', linewidth=1.5, linestyle='--', alpha=0.6, label='Chance (50%)')
ax.axvline(0.75, color='#0369a1', linewidth=1.0, linestyle=':', alpha=0.5, label='75% threshold')

# y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels([m[0] for m in reversed(models)], fontsize=9)

ax.set_xlabel("EB*-leads fraction (Wilson 95% CI)", fontweight='bold')
ax.set_title("C1-B: Temporal Precedence of Binding over Behavior\n(41-term canonical dataset, EB* leads = binding rises before behavior)",
             fontweight='bold', pad=12)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.7, len(models) - 0.3)
ax.grid(axis='x', alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_handles = [
    mpatches.Patch(color=FAMILY_COLORS['pythia'],  label='Pythia family'),
    mpatches.Patch(color=FAMILY_COLORS['olmo'],    label='OLMo-1B'),
    mpatches.Patch(color=FAMILY_COLORS['crfm'],    label='CRFM GPT-2 Sm (5 seeds)'),
    mpatches.Patch(color=FAMILY_COLORS['smollm3'], label='SmolLM3-3B (‡ censored)'),
    plt.Line2D([0], [0], color='#dc2626', linewidth=1.5, linestyle='--', label='Chance (50%)'),
]
ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9, fontsize=8.5)

# Footnote
fig.text(0.02, -0.02,
         "‡ SmolLM3 C1-B likely censored: earliest checkpoint already post-peak binding. "
         "Significance: *** p<0.001, ** p<0.01, ns p≥0.05, — p=1.000 (anti-leads).",
         fontsize=7.5, color='#555', style='italic')

plt.tight_layout()
fig.subplots_adjust(left=0.22)

output_dir = Path('paper/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'c1b_forest_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'c1b_forest_plot.pdf', bbox_inches='tight')

print("✅ Generated C1-B forest plot")
print(f"   paper/figures/c1b_forest_plot.png")
print(f"   paper/figures/c1b_forest_plot.pdf")
