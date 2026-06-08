"""Generate prompt_robustness_heatmap figure.

Heatmap of mean EB* across 9 terms × 6 generation prompts,
sorted by coefficient of variation (CV, low → high).
Data: merged_100.csv averaged across all models and checkpoints.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path

BASE    = Path('/workspaces/attention-binding-a11y')
OUT_DIR = BASE / 'paper/figures'

PIDS = ['gen_001', 'gen_002', 'gen_003', 'gen_004', 'gen_005', 'gen_006']

df    = pd.read_csv(BASE / 'data/results/merged_100.csv')
pivot = df.groupby(['term', 'prompt_id'])['eb_star'].mean().unstack()
pivot = pivot.reindex(columns=PIDS)
pivot['cv'] = pivot.std(axis=1) / pivot.mean(axis=1)
pivot = pivot.sort_values('cv')
cv_vals = pivot['cv'].copy()
heatmap  = pivot[PIDS]

fig, ax = plt.subplots(figsize=(11, 7))

cmap = mcolors.LinearSegmentedColormap.from_list(
    'rg', ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'])
im = ax.imshow(heatmap.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

# Cell value annotations
for i in range(len(heatmap)):
    for j in range(len(PIDS)):
        val = heatmap.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color='black' if val > 0.15 else 'white')

ax.set_xticks(range(len(PIDS)))
ax.set_xticklabels(PIDS, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(heatmap)))
ax.set_yticklabels(heatmap.index, fontsize=10)
ax.set_xlabel('Prompt ID', fontweight='bold', fontsize=11)
ax.set_ylabel('Accessibility Term', fontweight='bold', fontsize=11)
ax.set_title('Prompt Robustness: EB* Across 9 Terms × 6 Generation Prompts\n'
             '(Sorted by CV, low → high)',
             fontweight='bold', fontsize=12, pad=12)

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)

# CV annotations on colorbar — positioned to the RIGHT of the colorbar ticks
n = len(heatmap)
for i, (term, cv) in enumerate(cv_vals.items()):
    y_frac = 1.0 - (i + 0.5) / n
    weight = 'bold' if cv > 0.1 else 'normal'
    cbar.ax.text(3.2, y_frac, f'CV={cv:.3f}',
                 transform=cbar.ax.transAxes,
                 va='center', ha='left', fontsize=8, fontweight=weight)

# Colorbar label placed far right of CV annotations to avoid overlap
cbar.ax.text(6.5, 0.5, 'EB* (Emergent Binding)',
             transform=cbar.ax.transAxes,
             va='center', ha='left', fontsize=10, fontweight='bold',
             rotation=90, rotation_mode='anchor')

# Mean CV summary box
mean_cv = cv_vals.mean()
ax.text(0.99, 0.01, f'Mean CV = {mean_cv:.3f}\n(Low prompt sensitivity)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9c3',
                  edgecolor='#d97706', linewidth=1.2))

plt.tight_layout()
plt.savefig(OUT_DIR / 'prompt_robustness_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'prompt_robustness_heatmap.pdf', bbox_inches='tight')
plt.close()
print('Done: prompt_robustness_heatmap')
