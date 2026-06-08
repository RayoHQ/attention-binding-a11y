"""Generate Pythia lifecycle figures.

Produces four figures:
  Fig 1 — correlation_lifecycle.png       ρ(EB*, Beh) early vs late (hardcoded from C4-B summaries)
  Fig 2 — phase_transition_scatter.png    EB* vs Beh scatter early/late (6 panels, 3-term data)
  Fig 3 — term_heterogeneity_2b8.png      Per-term EB* + Beh trajectories at 2.8B (3-term data)
  Fig 4 — figure4_1b_decoupling.png       EB* and Beh dual-axis at 1B (hardcoded from paper §4.4)

Figs 1 & 4 use verified values from C4-B JSON summaries and paper §4.2/4.4.
Figs 2 & 3 load from data/results/binding/ and data/results/behavioral/ (3-term raw files).
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

BASE_DIR       = Path('/teamspace/studios/this_studio/attention-binding-a11y')
BINDING_DIR    = BASE_DIR / 'data/results/binding'
BEHAVIORAL_DIR = BASE_DIR / 'data/results/behavioral'
OUT_DIR        = BASE_DIR / 'paper/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = ['160m', '1b', '2.8b']
STEPS = [0, 15000, 30000, 60000, 90000, 120000, 140000, 143000]

SCALE_COLORS = {'160m': '#f59e0b', '1b': '#3b82f6', '2.8b': '#10b981'}
SCALE_LABELS = {'160m': 'Pythia-160M', '1b': 'Pythia-1B', '2.8b': 'Pythia-2.8B'}

# ── Hardcoded lifecycle data (verified from C4-B JSON summaries + paper §4.4) ─
# C4-B summaries: pythia-{size}_decoupling_summary.json
RHO_EARLY = {'160m': 0.479, '1b': 0.739, '2.8b': 0.613}
RHO_LATE  = {'160m': 0.044, '1b': -0.054, '2.8b': 0.270}

# 1B EB* and behavioral trajectories (from paper §4.2 / §4.4 text)
# EB* peaks at step 15k (0.646), plateaus 0.595-0.611 thereafter
# Behavioral rises from 0.167 (step 0) to 0.806 (step 143k), peaks 0.833 at step 30k
STEPS_1B = [0, 15, 30, 60, 90, 120, 140, 143]   # ×1000
EB_1B    = [0.480, 0.646, 0.611, 0.605, 0.601, 0.598, 0.596, 0.595]
BEH_1B   = [0.167, 0.444, 0.833, 0.750, 0.778, 0.778, 0.806, 0.806]


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mean_eb_per_term(size, step):
    """Return {term: mean_eb_star} from binding jsonl."""
    rows = load_jsonl(BINDING_DIR / f'{size}_step{step}_binding.jsonl')
    by_term = defaultdict(list)
    for r in rows:
        by_term[r['term']].append(r['eb_star'])
    return {t: np.mean(v) for t, v in by_term.items()}


def mean_beh_per_term(size, step):
    """Return {term: mean_score} from behavioral jsonl.
    Handles both raw (score field) and pre-aggregated (beh_avg field) formats."""
    rows = load_jsonl(BEHAVIORAL_DIR / f'{size}_step{step}_behavioral.jsonl')
    by_term = defaultdict(list)
    for r in rows:
        val = r.get('beh_avg', r.get('score'))
        if val is not None:
            by_term[r['term']].append(val)
    return {t: np.mean(v) for t, v in by_term.items()}


def build_lifecycle(size):
    """Return (steps_used, rhos, eb_means, beh_means) across checkpoints."""
    steps_used, rhos, eb_traj, beh_traj = [], [], [], []
    for step in STEPS:
        eb  = mean_eb_per_term(size, step)
        beh = mean_beh_per_term(size, step)
        terms = sorted(set(eb) & set(beh))
        if len(terms) < 2:
            continue
        eb_vals  = [eb[t]  for t in terms]
        beh_vals = [beh[t] for t in terms]
        rho, _ = spearmanr(eb_vals, beh_vals)
        steps_used.append(step)
        rhos.append(rho)
        eb_traj.append(np.mean(eb_vals))
        beh_traj.append(np.mean(beh_vals))
    return steps_used, rhos, eb_traj, beh_traj


print("Generating lifecycle figures...")


# ── Figure 1: Correlation lifecycle ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

LATE_OFFSETS = {'160m': (-42, 12), '1b': (-42, -16), '2.8b': (-42, 5)}

for size in SIZES:
    rho_e = RHO_EARLY[size]
    rho_l = RHO_LATE[size]
    ax.plot([15, 143], [rho_e, rho_l], 'o-', color=SCALE_COLORS[size],
            label=SCALE_LABELS[size], linewidth=2.5, markersize=9,
            markeredgecolor='white', markeredgewidth=1.2)
    ax.annotate(f'ρ={rho_e:+.2f}', (15, rho_e), fontsize=8, color=SCALE_COLORS[size],
                xytext=(2, 4), textcoords='offset points')
    ax.annotate(f'ρ={rho_l:+.2f}', (143, rho_l), fontsize=8, color=SCALE_COLORS[size],
                xytext=LATE_OFFSETS[size], textcoords='offset points')

ax.axhline(0, color='#6b7280', linewidth=1.2, linestyle='--', alpha=0.7)
ax.fill_between([0, 143], 0, 1,  alpha=0.04, color='#3b82f6')
ax.fill_between([0, 143], -1, 0, alpha=0.04, color='#ef4444')

ax.text(5, 0.62,  'Coupling\nphase',   fontsize=9, color='#3b82f6', style='italic')
ax.text(5, 0.04, 'Decoupling\nphase', fontsize=9, color='#ef4444', style='italic')

ax.set_xlabel('Training step (×1000)', fontweight='bold')
ax.set_ylabel('Spearman ρ (EB* vs Behavioral score)', fontweight='bold')
ax.set_title('C1/C4 Correlation Lifecycle: Early Coupling → Late Decoupling\n'
             '(verified mean Spearman ρ from C4-B population test, 41-term canonical dataset)',
             fontweight='bold', pad=10)
ax.set_xlim(0, 155)
ax.set_ylim(-0.25, 0.90)
ax.set_xticks([15, 143])
ax.set_xticklabels(['Early\n(step 15k)', 'Late\n(step 143k)'])
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / 'correlation_lifecycle.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'correlation_lifecycle.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 1: correlation_lifecycle")


# ── Figure 2: Phase transition scatter (6 panels) ────────────────────────────

EARLY_STEP = 15000
LATE_STEP  = 143000

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for col, size in enumerate(SIZES):
    for row, (step, phase, color, title_sfx) in enumerate([
        (EARLY_STEP, 'Early', '#1d4ed8', f'step {EARLY_STEP//1000}k'),
        (LATE_STEP,  'Late',  '#dc2626', f'step {LATE_STEP//1000}k'),
    ]):
        ax = axes[row][col]
        eb  = mean_eb_per_term(size, step)
        beh = mean_beh_per_term(size, step)
        terms = sorted(set(eb) & set(beh))
        eb_vals  = [eb[t]  for t in terms]
        beh_vals = [beh[t] for t in terms]
        rho, pval = spearmanr(eb_vals, beh_vals)

        ax.scatter(eb_vals, beh_vals, color=color, alpha=0.8, s=60,
                   edgecolors='white', linewidth=0.6, zorder=3)
        for i, t in enumerate(terms):
            ax.annotate(t, (eb_vals[i], beh_vals[i]),
                        fontsize=6.5, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color='#555')

        # Diagonal reference
        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], '--', color='#9ca3af', linewidth=0.9, alpha=0.5)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{SCALE_LABELS[size]}\n{phase} ({title_sfx})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('EB* (mean per term)', fontsize=9)
        ax.set_ylabel('Behavioral score', fontsize=9)
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        ax.text(0.05, 0.93, f'ρ={rho:+.2f} {sig}', transform=ax.transAxes,
                fontsize=9, color=color, fontweight='bold')
        ax.grid(alpha=0.15, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle('C1/C4 Phase Transition: EB* vs Behavioral Score at Early and Late Checkpoints\n'
             '(each point = one term, n=9 terms per panel)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'phase_transition_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'phase_transition_scatter.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 2: phase_transition_scatter")


# ── Figure 3: Term heterogeneity at 2.8B ─────────────────────────────────────

size = '2.8b'
eb_by_term  = {}
beh_by_term = {}
for step in STEPS:
    eb  = mean_eb_per_term(size, step)
    beh = mean_beh_per_term(size, step)
    for t in eb:
        eb_by_term.setdefault(t, {})[step] = eb[t]
    for t in beh:
        beh_by_term.setdefault(t, {})[step] = beh.get(t, np.nan)

terms_28b = sorted(set(eb_by_term) & set(beh_by_term))
x_steps = [s / 1000 for s in STEPS]

cmap = plt.get_cmap('tab10')
term_colors = {t: cmap(i % 10) for i, t in enumerate(terms_28b)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for t in terms_28b:
    eb_vals  = [eb_by_term[t].get(s, np.nan)  for s in STEPS]
    beh_vals = [beh_by_term[t].get(s, np.nan) for s in STEPS]
    ax1.plot(x_steps, eb_vals,  'o-', color=term_colors[t], linewidth=1.6,
             markersize=4, alpha=0.85, label=t)
    ax2.plot(x_steps, beh_vals, 'o-', color=term_colors[t], linewidth=1.6,
             markersize=4, alpha=0.85, label=t)

for ax, ylabel, title in [
    (ax1, 'Mean EB*',           'EB* Trajectories (Pythia-2.8B)'),
    (ax2, 'Behavioral score',   'Behavioral Trajectories (Pythia-2.8B)'),
]:
    ax.set_xlabel('Training step (×1000)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlim(-2, 148)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.18, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.85)
fig.suptitle('C4 Term Heterogeneity at 2.8B Scale: Binding and Behavior Develop Independently\n'
             '(9 terms, each line = one term)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'term_heterogeneity_2b8.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'term_heterogeneity_2b8.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 3: term_heterogeneity_2b8")


# ── Figure 4: 1B decoupling dual-axis ────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

line1, = ax1.plot(STEPS_1B, EB_1B,  'o-', color='#dc2626', linewidth=2.5, markersize=8,
                  markeredgecolor='white', markeredgewidth=1.2, label='EB* (binding strength)')
line2, = ax2.plot(STEPS_1B, BEH_1B, 's-', color='#16a34a', linewidth=2.5, markersize=8,
                  markeredgecolor='white', markeredgewidth=1.2, label='Behavioral score')

# Shade decoupling region (after EB* peak at step 15k)
ax1.axvspan(15, 143, alpha=0.06, color='#6b7280',
            label='Decoupling period (post step 15k)')

ax1.set_xlabel('Training step (×1000)', fontweight='bold')
ax1.set_ylabel('Mean EB* (binding strength)', color='#dc2626', fontweight='bold')
ax2.set_ylabel('Behavioral score (rec + gen mean)', color='#16a34a', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#dc2626')
ax2.tick_params(axis='y', labelcolor='#16a34a')

ax1.set_title('C4 Decoupling at 1B Scale: EB* Saturates While Behavior Continues Rising\n'
              '(Pythia-1B-deduped, 9-term mean across checkpoints)',
              fontweight='bold', pad=10)
ax1.set_xlim(-2, 150)
ax1.grid(alpha=0.18, linestyle='--')
ax1.spines['top'].set_visible(False)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'figure4_1b_decoupling.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'figure4_1b_decoupling.pdf', bbox_inches='tight')
plt.close()
print("✅ Fig 4: figure4_1b_decoupling")

print("\nAll 4 lifecycle figures saved to paper/figures/")
