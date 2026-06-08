"""Generate Slide 1 illustration: Behavioral Black Box vs. Mechanistic Monitoring.

Outputs:
  paper/figures/slide_blackbox_vs_mechanistic.png
  paper/figures/slide_blackbox_vs_mechanistic.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

# --- Style (matches repo conventions) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

OUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Colours ---
GREY       = '#CCCCCC'
DARK_GREY  = '#555555'
BLUE       = '#2171B5'
ORANGE     = '#E6550D'
GREEN      = '#238B45'
BG_LEFT    = '#F7F7F7'
BG_RIGHT   = '#EFF6FF'


def draw_arrow(ax, x0, y0, x1, y1, color='#333333', lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def rounded_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=10,
                text_color='white', bold=False, va='center'):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle='round,pad=0.05',
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va=va, fontsize=fontsize,
            color=text_color, weight=weight, wrap=True,
            multialignment='center')


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('white')

for ax in (ax_l, ax_r):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

# ===== LEFT PANEL: Behavioral Evaluation =====
ax_l.add_patch(plt.Rectangle((0, 0), 10, 10, facecolor=BG_LEFT, edgecolor='#BBBBBB', lw=1.2))
ax_l.text(5, 9.3, 'Behavioral Evaluation', ha='center', va='center',
          fontsize=13, fontweight='bold', color=DARK_GREY)
ax_l.text(5, 8.7, '(what models can do)', ha='center', va='center',
          fontsize=9.5, color='#888888', style='italic')

# Input
rounded_box(ax_l, 5, 7.5, 4.5, 0.9,
            '"What is a screen reader?"',
            facecolor='#E8E8E8', edgecolor=DARK_GREY, fontsize=9.5,
            text_color=DARK_GREY, bold=False)

draw_arrow(ax_l, 5, 7.05, 5, 6.35, color=DARK_GREY)

# Black box
rounded_box(ax_l, 5, 5.8, 4.5, 1.0,
            '[  ]  Language Model\n(opaque internals)',
            facecolor='#444444', edgecolor='#222222', fontsize=10,
            text_color='white', bold=True)

draw_arrow(ax_l, 5, 5.3, 5, 4.55, color=DARK_GREY)

# Output: only accuracy
rounded_box(ax_l, 5, 4.1, 4.5, 0.85,
            'Accuracy: 72%  ✓ / ✗',
            facecolor='#E8E8E8', edgecolor=DARK_GREY, fontsize=9.5,
            text_color=DARK_GREY)

# Pain point annotation
ax_l.add_patch(FancyBboxPatch((0.4, 0.3), 9.2, 2.7,
                               boxstyle='round,pad=0.1',
                               facecolor='#FDECEA', edgecolor='#CC4444', lw=1.2, ls='--'))
ax_l.text(5, 2.9, '✗  No insight into', ha='center', va='center',
          fontsize=9.5, color='#CC4444', fontweight='bold')
ax_l.text(5, 2.4, 'when internal representations form', ha='center', va='center',
          fontsize=9, color='#CC4444')
ax_l.text(5, 1.9, '✗  Expensive: run at every checkpoint', ha='center', va='center',
          fontsize=9, color='#CC4444')
ax_l.text(5, 1.4, '✗  Binary signal: pass or fail', ha='center', va='center',
          fontsize=9, color='#CC4444')

# ===== RIGHT PANEL: Mechanistic Monitoring =====
ax_r.add_patch(plt.Rectangle((0, 0), 10, 10, facecolor=BG_RIGHT, edgecolor='#7BAFD4', lw=1.2))
ax_r.text(5, 9.3, 'Mechanistic Monitoring (EB*)', ha='center', va='center',
          fontsize=13, fontweight='bold', color=BLUE)
ax_r.text(5, 8.7, '(how & when representations form)', ha='center', va='center',
          fontsize=9.5, color='#888888', style='italic')

# Input  — mirrors left panel exactly
rounded_box(ax_r, 5, 7.5, 4.5, 0.9,
            '"What is a screen reader?"',
            facecolor='#D6E8F5', edgecolor=BLUE, fontsize=9.5,
            text_color=DARK_GREY, bold=False)

draw_arrow(ax_r, 5, 7.05, 5, 6.35, color=BLUE)

# Model box
rounded_box(ax_r, 5, 5.8, 4.5, 1.0,
            '[o]  Language Model\n(attention heads visible)',
            facecolor=BLUE, edgecolor='#1351A0', fontsize=10,
            text_color='white', bold=True)

draw_arrow(ax_r, 5, 5.3, 5, 4.525, color=BLUE)

# EB* output box — same geometry as left output box (y=4.1, h=0.85)
rounded_box(ax_r, 5, 4.1, 4.5, 0.85,
            'EB* = 0.64  →  Binding detected',
            facecolor='#D6E8F5', edgecolor=BLUE, fontsize=9.5,
            text_color=DARK_GREY)

# Early-warning callout — compact text in the gap (y 3.675 to 3.3)
ax_r.text(5, 3.52, 'Detected at step 15K \u2014 behavior score still at 0.20',
          ha='center', va='center', fontsize=8.5, color=BLUE, style='italic',
          bbox=dict(facecolor='#D6E8F5', edgecolor=BLUE,
                    boxstyle='round,pad=0.18', lw=1.0, alpha=0.85))

# Benefits box — same geometry as red box: (0.4, 0.3), w=9.2, h=2.7
ax_r.add_patch(FancyBboxPatch((0.4, 0.3), 9.2, 2.7,
                               boxstyle='round,pad=0.1',
                               facecolor='#E6F4EA', edgecolor='#2D8A4E', lw=1.2, ls='--'))
# Text at same y-positions as red box
ax_r.text(5, 2.9, '✓  Lightweight early-warning signal', ha='center', va='center',
          fontsize=9.5, color='#2D8A4E', fontweight='bold')
ax_r.text(5, 2.4, 'EB* peaks in the first 10% of training', ha='center', va='center',
          fontsize=9, color='#2D8A4E')
ax_r.text(5, 1.9, '✓  Forward-pass only — no auxiliary training', ha='center', va='center',
          fontsize=9, color='#2D8A4E')
ax_r.text(5, 1.4, '✓  Scales to 405B+, GQA-compatible, ~1 MB per pass', ha='center', va='center',
          fontsize=9, color='#2D8A4E')

# Divider
fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.97],
                           transform=fig.transFigure,
                           color='#AAAAAA', lw=1.5, ls='-'))

plt.tight_layout(pad=1.5)

for ext in ('png', 'pdf'):
    out = OUT_DIR / f'slide_blackbox_vs_mechanistic.{ext}'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')

plt.close()
