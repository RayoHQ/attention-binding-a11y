"""Generate format_diversity_analysis figure.

Bar chart (left) and violin plot (right) of EB* by prompt format type.
Data source: data/results/merged_100.csv (generation prompts only, n=1296).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

BASE_DIR = Path('/workspaces/attention-binding-a11y')
OUT_DIR  = BASE_DIR / 'paper/figures'

FORMAT_MAP = {
    'gen_001': 'Definition',
    'gen_006': 'Best Practice',
    'gen_003': 'User Benefit',
    'gen_004': 'Implementation',
    'gen_005': 'Failure Case',
    'gen_002': 'Context/Tutorial',
}
FORMAT_ORDER = ['Definition', 'Best Practice', 'User Benefit',
                'Implementation', 'Failure Case', 'Context/Tutorial']

PALETTE = ['#6366f1', '#4f8fc0', '#38b2ac', '#48bb78', '#a0c878', '#d4e157']

df = pd.read_csv(BASE_DIR / 'data/results/merged_100.csv')
df = df[df['task'] == 'generation'].copy()
df['format_type'] = df['prompt_id'].map(FORMAT_MAP)
df = df[df['format_type'].notna()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ── Left: bar chart ───────────────────────────────────────────────────────────
means = df.groupby('format_type')['eb_star'].mean().reindex(FORMAT_ORDER)
stds  = df.groupby('format_type')['eb_star'].std().reindex(FORMAT_ORDER)
ns    = df.groupby('format_type')['eb_star'].count().reindex(FORMAT_ORDER)

bars = ax1.bar(FORMAT_ORDER, means, yerr=stds, capsize=4,
               color=PALETTE, edgecolor='white', linewidth=0.8,
               error_kw={'linewidth': 1.2, 'ecolor': '#374151'})

overall_mean = df['eb_star'].mean()
ax1.axhline(overall_mean, color='#dc2626', linewidth=1.5, linestyle='--',
            label=f'Overall mean = {overall_mean:.3f}')

for bar, n, mean in zip(bars, ns, means):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[FORMAT_ORDER[list(means).index(mean)]] + 0.02,
             f'n={int(n)}', ha='center', va='bottom', fontsize=8, color='#374151')

ax1.set_ylim(0, 1.0)
ax1.set_ylabel('Mean EB* ± Std Dev', fontweight='bold')
ax1.set_xlabel('Prompt Format Type', fontweight='bold')
ax1.set_title('Prompt Format Diversity: EB* Across 6 Generation Types\nPattern Holds Across All Formats',
              fontweight='bold', pad=10)
ax1.set_xticklabels(FORMAT_ORDER, rotation=45, ha='right', fontsize=9)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(axis='y', alpha=0.2, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Right: violin plot ────────────────────────────────────────────────────────
violin_data = [df[df['format_type'] == ft]['eb_star'].values for ft in FORMAT_ORDER]

parts = ax2.violinplot(violin_data, positions=range(len(FORMAT_ORDER)),
                       showmeans=True, showmedians=True, showextrema=True)

for i, (body, color) in enumerate(zip(parts['bodies'], PALETTE)):
    body.set_facecolor(color)
    body.set_alpha(0.7)
    body.set_edgecolor('none')

parts['cmeans'].set_color('#1e40af')
parts['cmeans'].set_linewidth(1.5)
parts['cmedians'].set_color('#1e40af')
parts['cmedians'].set_linewidth(2)
parts['cmedians'].set_linestyle('-')
parts['cbars'].set_color('#1e40af')
parts['cbars'].set_linewidth(1)
parts['cmins'].set_color('#1e40af')
parts['cmins'].set_linewidth(1)
parts['cmaxes'].set_color('#1e40af')
parts['cmaxes'].set_linewidth(1)

ax2.set_ylim(0, 1.0)
ax2.set_ylabel('EB* Distribution', fontweight='bold')
ax2.set_xlabel('Prompt Format Type', fontweight='bold')
ax2.set_title('EB* Distribution by Prompt Format\nConsistent Variance Across Types',
              fontweight='bold', pad=10)
ax2.set_xticks(range(len(FORMAT_ORDER)))
ax2.set_xticklabels(FORMAT_ORDER, rotation=45, ha='right', fontsize=9)
ax2.grid(axis='y', alpha=0.2, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

median_patch = mpatches.Patch(color='none', label='Median')
mean_patch   = mpatches.Patch(color='none', label='Mean')
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1e40af', linewidth=2, linestyle='-',  label='Median'),
    Line2D([0], [0], color='#1e40af', linewidth=1.5, linestyle='--', label='Mean'),
]
ax2.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# Statistics box — placed in upper-right to avoid overlapping rotated x-axis labels
std_overall = df['eb_star'].std()
cv = std_overall / overall_mean
stats_text = (f'Overall Statistics:\n'
              f'Mean EB* = {overall_mean:.3f}\n'
              f'Std Dev = {std_overall:.3f}\n'
              f'CV = {cv:.3f}\n'
              f'Formats = {len(FORMAT_ORDER)}\n'
              f'Total n = {len(df)}')
ax2.text(0.97, 0.97, stats_text,
         transform=ax2.transAxes,
         fontsize=8, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9c3',
                   edgecolor='#d1d5db', alpha=0.9))

plt.tight_layout(pad=2.0)
plt.savefig(OUT_DIR / 'format_diversity_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'format_diversity_analysis.pdf', bbox_inches='tight')
plt.close()
print('Done: format_diversity_analysis')
