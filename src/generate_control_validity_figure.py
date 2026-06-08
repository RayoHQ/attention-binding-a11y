"""Generate discriminant validity figure showing V2/V3/V4 control gradient."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Data from V2/V3/V4 experiments
controls_data = [
    # V2 controls (both scales same)
    {"name": "True\nNonsense", "eb_star_160m": 0.26, "eb_star_1b": 0.26, "type": "v2", "color": "#e74c3c"},
    {"name": "Cross-\nLanguage", "eb_star_160m": 0.41, "eb_star_1b": 0.41, "type": "v2", "color": "#e67e22"},
    {"name": "Rare\nPairs", "eb_star_160m": 0.50, "eb_star_1b": 0.50, "type": "v2", "color": "#f39c12"},
    
    # Real terms (baseline)
    {"name": "Real\nTerms", "eb_star_160m": 0.74, "eb_star_1b": 0.74, "type": "real", "color": "#27ae60"},
    
    # V3/V4 irrelevant controls (scale-dependent)
    {"name": "V3/V4\nIrrelevant\n(160M)", "eb_star_160m": 0.861, "eb_star_1b": None, "type": "v34_160m", "color": "#9b59b6"},
    {"name": "V3/V4\nIrrelevant\n(1B)", "eb_star_160m": None, "eb_star_1b": 0.639, "type": "v34_1b", "color": "#3498db"},
]

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Bar chart showing gradient at both scales
x_pos = np.arange(len(controls_data))
width = 0.35

for i, ctrl in enumerate(controls_data):
    # 160M bars
    if ctrl["eb_star_160m"] is not None:
        ax1.bar(i - width/2, ctrl["eb_star_160m"], width, 
                color=ctrl["color"], alpha=0.7, edgecolor='black', linewidth=1)
    
    # 1B bars
    if ctrl["eb_star_1b"] is not None:
        ax1.bar(i + width/2, ctrl["eb_star_1b"], width,
                color=ctrl["color"], alpha=1.0, edgecolor='black', linewidth=1)

# Add real terms reference line
ax1.axhline(y=0.74, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7, label='Real Terms Baseline')

# Annotate the failure/success
ax1.text(4.3, 0.90, 'Cannot\nDiscriminate', fontsize=9, ha='center', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.3))
ax1.text(5.3, 0.68, 'Partial\nDiscrimination', fontsize=9, ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.3))

ax1.set_ylabel('Mean EB* Score', fontweight='bold')
ax1.set_xlabel('Control Type', fontweight='bold')
ax1.set_title('A. Discriminant Validity Gradient Across Control Types', fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([c["name"] for c in controls_data], fontsize=8)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Legend for scales
legend_elements = [
    mpatches.Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='160M (Pythia-160M-deduped)'),
    mpatches.Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='1B (Pythia-1B-deduped)'),
]
ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

# Panel 2: Scale-dependent trajectory for V3/V4 irrelevant terms
scales = ['160M\nstep120k', '1B\nstep143k']
irrelevant_eb = [0.861, 0.639]
real_eb = [0.74, 0.74]

x_scale = [0, 1]

# Plot lines
ax2.plot(x_scale, irrelevant_eb, 'o-', color='#9b59b6', linewidth=3, 
         markersize=10, label='V3/V4 Irrelevant Controls', markeredgecolor='black', markeredgewidth=1.5)
ax2.plot(x_scale, real_eb, 's--', color='#27ae60', linewidth=2, 
         markersize=9, label='Real Terms (Baseline)', markeredgecolor='black', markeredgewidth=1.5)

# Shade discrimination regions
ax2.axhspan(0.74, 1.0, alpha=0.15, color='#e74c3c', label='Cannot Discriminate')
ax2.axhspan(0, 0.74, alpha=0.15, color='#3498db', label='Partial Discrimination')

# Annotate deltas
ax2.annotate('', xy=(0, 0.861), xytext=(0, 0.74),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(0.1, 0.80, '+0.121', fontsize=9, color='red', fontweight='bold')

ax2.annotate('', xy=(1, 0.74), xytext=(1, 0.639),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax2.text(1.1, 0.69, '−0.101', fontsize=9, color='blue', fontweight='bold')

ax2.set_ylabel('Mean EB* Score', fontweight='bold')
ax2.set_xlabel('Model Scale', fontweight='bold')
ax2.set_title('B. Scale-Dependent Discrimination Pattern', fontweight='bold', pad=15)
ax2.set_xticks(x_scale)
ax2.set_xticklabels(scales)
ax2.set_ylim(0.5, 0.95)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)

plt.tight_layout()

# Save figure
output_dir = Path('paper/figures')
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / 'discriminant_validity_controls.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'discriminant_validity_controls.pdf', bbox_inches='tight')

print("✅ Generated discriminant validity control figure")
print(f"   - {output_dir / 'discriminant_validity_controls.png'}")
print(f"   - {output_dir / 'discriminant_validity_controls.pdf'}")

