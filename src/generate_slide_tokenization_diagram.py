"""Generate Slide 2 illustration: Tokenization + Attention Span + BSI formula.

Two-panel figure:
  Top:    "screen reader" → tokenizer → [screen][reader] span with bigram highlight
  Bottom: 5x5 attention matrix heatmap with within-span cells highlighted,
          annotated with the BSI / EB* formula

Outputs:
  paper/figures/slide_tokenization_attention.png
  paper/figures/slide_tokenization_attention.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

# --- Style ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

OUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE       = '#2171B5'
ORANGE     = '#E6550D'
ORANGE_L   = '#FDCBA0'
GREEN      = '#238B45'
GREY       = '#555555'
LIGHT_BLUE = '#D6E8F5'
SPAN_COLOR = '#FF7F0E'

# ============================================================
# Figure layout: 2 rows, with the bottom split (matrix + formula)
# ============================================================
fig = plt.figure(figsize=(13, 8), facecolor='white')
gs  = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 2.2], hspace=0.35, wspace=0.3)

ax_top    = fig.add_subplot(gs[0, :])   # full-width tokenisation diagram
ax_matrix = fig.add_subplot(gs[1, 0])   # attention matrix
ax_formula = fig.add_subplot(gs[1, 1])  # formula / legend

for ax in (ax_top, ax_matrix, ax_formula):
    ax.axis('off')

# ============================================================
# TOP PANEL — Tokenisation diagram
# ============================================================
ax_top.set_xlim(0, 14)
ax_top.set_ylim(0, 4)
ax_top.set_aspect('auto')
ax_top.set_title('Step 1: Multi-Token Term → Attention Span',
                 fontsize=13, fontweight='bold', color=GREY, pad=8)

# Raw text bubble
bubble = FancyBboxPatch((0.3, 1.3), 2.5, 1.4,
                         boxstyle='round,pad=0.15',
                         facecolor='#F0F0F0', edgecolor=GREY, lw=1.5)
ax_top.add_patch(bubble)
ax_top.text(1.55, 2.35, 'Input text', ha='center', fontsize=8.5, color=GREY, style='italic')
ax_top.text(1.55, 1.85, '"screen reader"', ha='center', fontsize=11,
            color='#222222', fontweight='bold')

# Arrow → Tokenizer
ax_top.annotate('', xy=(4.0, 2.0), xytext=(2.9, 2.0),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.8))

# Tokenizer box
tok_box = FancyBboxPatch((4.1, 1.3), 2.2, 1.4,
                          boxstyle='round,pad=0.15',
                          facecolor='#E8F4E8', edgecolor=GREEN, lw=1.5)
ax_top.add_patch(tok_box)
ax_top.text(5.2, 2.35, 'Tokenizer', ha='center', fontsize=8.5, color=GREEN, style='italic')
ax_top.text(5.2, 1.85, '⚙ GPT-2 BPE', ha='center', fontsize=10, color=GREEN, fontweight='bold')

# Arrow → Tokens
ax_top.annotate('', xy=(7.1, 2.0), xytext=(6.4, 2.0),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.8))

# Token boxes with span highlight
tokens_display = ['[BOS]', 'screen', 'reader', '[EOS]']
colors_tok     = ['#E0E0E0', ORANGE_L, ORANGE_L, '#E0E0E0']
edge_colors    = [GREY, ORANGE, ORANGE, GREY]
x_starts = [7.2, 8.55, 9.9, 11.25]
w_tok = 1.2
for i, (tok, fc, ec, xs) in enumerate(zip(tokens_display, colors_tok, edge_colors, x_starts)):
    b = FancyBboxPatch((xs, 1.45), w_tok, 1.1,
                        boxstyle='round,pad=0.08',
                        facecolor=fc, edgecolor=ec, lw=2.0 if ec == ORANGE else 1.2)
    ax_top.add_patch(b)
    ax_top.text(xs + w_tok / 2, 2.0, tok,
                ha='center', va='center', fontsize=10,
                color='#111111', fontweight='bold' if ec == ORANGE else 'normal')
    ax_top.text(xs + w_tok / 2, 1.1, f'idx {i}',
                ha='center', va='center', fontsize=7.5, color=GREY)

# Span brace
ax_top.annotate('', xy=(11.35, 0.95), xytext=(8.65, 0.95),
                arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2.0))
ax_top.text(10.0, 0.6, 'Binding span  S = {1, 2}', ha='center',
            fontsize=9.5, color=ORANGE, fontweight='bold')

# Token indices labels
ax_top.text(7.8, 3.4, 'Non-span', ha='center', fontsize=8, color=GREY)
ax_top.text(10.0, 3.4, 'Within-span  (EB* measured here)', ha='center',
            fontsize=8.5, color=ORANGE, fontweight='bold')
ax_top.text(12.8, 3.4, 'Non-span', ha='center', fontsize=8, color=GREY)

# Arrow markers for span
for xs, xt in [(8.65, 9.15), (11.35, 12.65)]:
    ax_top.annotate('', xy=(xs + 0.6, 2.6), xytext=(xt, 3.1),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))


# ============================================================
# BOTTOM-LEFT — Attention matrix heatmap
# ============================================================
ax_matrix.set_title('Step 2: Attention Matrix — One Head, One Layer',
                    fontsize=11.5, fontweight='bold', color=GREY, pad=8)
ax_matrix.axis('on')
ax_matrix.set_facecolor('white')

labels = ['[BOS]', 'screen', 'reader', '[EOS]']
n = len(labels)

# Simulated attention matrix (causal, softmaxed)
attn = np.array([
    [0.92, 0.00, 0.00, 0.00],   # [BOS] → only itself
    [0.30, 0.70, 0.00, 0.00],   # screen → BOS + itself
    [0.05, 0.68, 0.27, 0.00],   # reader → screen (high! binding)
    [0.08, 0.15, 0.62, 0.15],   # [EOS]  → distributes
])

# Mask within-span: rows [1,2] x cols [0..current] with span cells highlighted
mask_span = np.zeros((n, n), dtype=bool)
mask_span[2, 1] = True  # reader attends to screen  ← the key BSI cell

# Plot base heatmap (light blue)
cmap_base = plt.cm.Blues
im = ax_matrix.imshow(attn, cmap=cmap_base, vmin=0, vmax=1, aspect='auto')

# Overlay the span cell in orange
span_overlay = np.ma.masked_where(~mask_span, attn)
ax_matrix.imshow(span_overlay, cmap=plt.cm.Oranges, vmin=0, vmax=1, aspect='auto')

# Cell annotations
for i in range(n):
    for j in range(n):
        val = attn[i, j]
        if val == 0.0:
            continue
        color = 'white' if val > 0.5 else '#222222'
        weight = 'bold' if mask_span[i, j] else 'normal'
        ax_matrix.text(j, i, f'{val:.2f}',
                       ha='center', va='center', fontsize=9.5,
                       color=color, fontweight=weight)

# Draw orange border around span cells
for (i, j) in zip(*np.where(mask_span)):
    ax_matrix.add_patch(mpatches.Rectangle(
        (j - 0.5, i - 0.5), 1, 1,
        linewidth=3, edgecolor=ORANGE, facecolor='none'))

ax_matrix.set_xticks(range(n))
ax_matrix.set_yticks(range(n))
ax_matrix.set_xticklabels(labels, fontsize=10)
ax_matrix.set_yticklabels(labels, fontsize=10)
ax_matrix.set_xlabel('Source token (attends from)', fontsize=10, color=GREY)
ax_matrix.set_ylabel('Destination token (attends to)', fontsize=10, color=GREY)

# Orange label for the key cell
ax_matrix.text(1, 2.0, '← BSI cell\n(reader→screen)', ha='center',
               fontsize=8, color=ORANGE, fontweight='bold',
               bbox=dict(facecolor='white', edgecolor=ORANGE, boxstyle='round,pad=0.2', lw=1.2))

cbar = plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)
cbar.set_label('Attention weight', fontsize=9)

# ============================================================
# BOTTOM-RIGHT — Formula explanation
# ============================================================
ax_formula.set_xlim(0, 10)
ax_formula.set_ylim(0, 10)
ax_formula.set_title('Step 3: EB* Calculation', fontsize=11.5, fontweight='bold',
                     color=GREY, pad=8)

def fbox(ax, x, y, w, h, txt, fc, ec, fs=10, tc='#222222', bold=False):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle='round,pad=0.15',
                        facecolor=fc, edgecolor=ec, lw=1.5)
    ax.add_patch(b)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal', multialignment='center')

# BSI definition
fbox(ax_formula, 5, 9.0, 9.0, 1.1,
     'BSI(T, \u2113, h)  =  mean attention(dest\u2192src)\nfor dest, src \u2208 S,  dest > src',
     fc=LIGHT_BLUE, ec=BLUE, fs=9.5, tc='#1A3A5C')

ax_formula.text(5, 8.00, '(average later-to-earlier within-span attention for head h at layer ℓ)',
                ha='center', fontsize=8.5, color=GREY, style='italic')

ax_formula.annotate('', xy=(5, 7.52), xytext=(5, 7.78),
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.6))

# EB per layer — center=6.90, h=1.2  → [6.30, 7.50]  (wider gap above, 0.65 below)
fbox(ax_formula, 5, 6.90, 9.0, 1.2,
     'EB(T, ℓ)  =  max_h BSI(T, ℓ, h)  −  mean_h BSI(T, ℓ, h)',
     fc='#EDF5E8', ec=GREEN, fs=9.5, tc='#1A3A1A')
ax_formula.text(5, 6.45, '(prominence of the single most binding head vs. layer average)',
                ha='center', fontsize=8.5, color='#3A6A3A', style='italic')

ax_formula.annotate('', xy=(5, 5.37), xytext=(5, 6.28),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6))

# EB* aggregate — center=4.75, h=1.2  → [4.15, 5.35]  (gap=0.95 above and below)
fbox(ax_formula, 5, 4.75, 9.0, 1.2,
     'EB*(T)  =  max_ℓ  EB(T, ℓ)',
     fc='#FFF3E0', ec=ORANGE, fs=11, tc='#7A2B00', bold=True)
ax_formula.text(5, 4.30, '(maximum binding prominence across all layers)',
                ha='center', fontsize=8.5, color='#7A4A00', style='italic')

# Key properties — shifted down 0.90 so gap orange→grey = 0.95
ax_formula.add_patch(FancyBboxPatch((0.3, -0.40), 9.4, 3.6,
                                     boxstyle='round,pad=0.1',
                                     facecolor='#F8F8F8', edgecolor='#AAAAAA', lw=1.0))
ax_formula.text(5, 2.90, 'Why EB* is lightweight', ha='center',
                fontsize=10, fontweight='bold', color=GREY)

props = [
    'Forward-pass only — no training required',
    'Reads 2–4 attention cells, not full seq-len matrix',
    '~1 MB per pass vs. TBs for full attention at 405B',
    'Compatible with GQA (per query-head)',
]
for i, txt in enumerate(props):
    ax_formula.text(5, 2.25 - i * 0.65, f'\u2713  {txt}', ha='center', fontsize=9,
                    color=GREEN, va='center', fontweight='bold')

plt.suptitle('Attention-Head Binding (EB*): From Token Span to Mechanistic Signal',
             fontsize=14, fontweight='bold', color='#1A1A2E', y=1.01)

for ext in ('png', 'pdf'):
    out = OUT_DIR / f'slide_tokenization_attention.{ext}'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')

plt.close()
