#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py -- the three figures for the basin paper. Reads stored results only.

Palette #1668B8 / #C4561A passes the six-check categorical validator (chroma floor,
CVD separation dE 23.2 protan, normal-vision dE 30.1, contrast >= 3:1). Basin identity
is carried by MARKER SHAPE as well as hue, so the figures survive greyscale printing
and colour-vision deficiency without relying on colour alone.
"""
import glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', 'Paper2', 'code'))
OUT = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT, exist_ok=True)

HIGH, LOW = '#1668B8', '#C4561A'
GRID, INK, MUTED = '#D8D8D8', '#1A1A1A', '#5A5A5A'
THRESH = 27.0

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.edgecolor': MUTED, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'figure.dpi': 400,
})

D = pd.read_csv(os.path.join(CODE, 'results', 'wpwr_adjudication_runs.csv'))
D['low'] = D.pt < THRESH


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  wrote {name}.pdf / .png')


# ---------------------------------------------------------------- Fig 1 (2-col)
fig, ax = plt.subplots(figsize=(7.16, 2.5))
arms = sorted(D.w.unique())
ax.axvspan(25, 28, color='#F2F2F2', zorder=0)
ax.axvline(THRESH, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
rng = np.random.default_rng(7)
for i, w in enumerate(arms):
    a = D[D.w == w]
    for mask, col, mk, ms in ((a.low, LOW, '^', 16), (~a.low, HIGH, 'o', 15)):
        s = a[mask]
        ax.scatter(s.pt, np.full(len(s), i) + rng.uniform(-.13, .13, len(s)),
                   s=ms, c=col, marker=mk, linewidths=0.5, edgecolors='white', zorder=3)
    ax.text(21.4, i, f'{len(a[a.low])}/20', ha='right', va='center',
            fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(arms)))
ax.set_yticklabels([('0.15\n(reported)' if w == 0.15 else f'{w:g}') for w in arms])
ax.set_ylim(-0.6, len(arms) - 0.4)
ax.set_xlim(20.4, 30.9)
ax.set_ylabel('reward transmit-power weight $W_{\\mathrm{pwr}}$')
ax.set_xlabel('mean transmit power over held-out evaluation (dBm)')
ax.text(26.5, len(arms) - 0.42, 'no run, at any weight', ha='center', va='bottom',
        fontsize=7.5, color=MUTED, style='italic')
ax.text(21.4, len(arms) - 0.45, 'low / 20', ha='right', va='bottom',
        fontsize=7, color=MUTED)
ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], ls='', marker='^', color=LOW, ms=4, label='low-power basin'),
                   Line2D([], [], ls='', marker='o', color=HIGH, ms=4, label='high-power basin')],
          loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
          handletextpad=0.3, columnspacing=1.6)
save(fig, 'fig1_basins_by_weight')

# ---------------------------------------------------------------- Fig 2 (1-col)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.axvspan(25, 28, color='#F2F2F2', zorder=0)
for mask, col, mk, lab in ((D.low, LOW, '^', 'low-power basin'),
                           (~D.low, HIGH, 'o', 'high-power basin')):
    s = D[mask]
    ax.scatter(s.pt, s.rlf, s=15, c=col, marker=mk, linewidths=0.5,
               edgecolors='white', zorder=3, label=lab)
ax.set_yscale('log')
ax.set_xlabel('mean transmit power (dBm)')
ax.set_ylabel('sustained radio-link failure (%)')
ax.set_xlim(22.4, 30.8)
ax.grid(color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
lo, hi = D[D.low].rlf, D[~D.low].rlf
ax.annotate(f'{lo.mean():.2f}%', (23.3, lo.mean()), xytext=(6, 8),
            textcoords='offset points', fontsize=7.5, color=LOW)
ax.annotate(f'{hi.mean():.2f}%', (28.55, hi.mean()), xytext=(0, 0),
            textcoords='offset points', fontsize=7.5, color=HIGH, ha='right', va='center')
ax.legend(loc='lower left', frameon=False, handletextpad=0.3, borderpad=0.1)
save(fig, 'fig2_rlf_vs_power')

# ---------------------------------------------------------------- Fig 3 (1-col)
# The job here is a 2x2 count, not a scatter: within a rail the exact position
# carries no information, and four runs overlapped to a single dot when plotted.
rows = []
for f in sorted(glob.glob(os.path.join(CODE, 'models', 'basin_v2', 'model_s*_metrics.csv'))):
    s_ = int(re.search(r'model_s(\d+)_', os.path.basename(f)).group(1))
    r = pd.read_csv(f).iloc[0]
    rows.append(dict(seed=s_, pt=float(r['pt_dbm']), rb=float(r['rb'])))
B = pd.DataFrame(rows)
B['plow'] = B.pt < THRESH
B['blow'] = B.rb < 40.0

cells = {}
for pl in (True, False):
    for bl in (True, False):
        sel = B[(B.plow == pl) & (B.blow == bl)]
        cells[(pl, bl)] = sorted(int(v) for v in sel.seed)
Kp, Kb, N = int(B.plow.sum()), int(B.blow.sum()), len(B)

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis('off')
for (pl, bl), seeds in cells.items():
    cx = 0.5 if bl else 1.5
    cy = 0.5 if pl else 1.5
    exp = (Kp if pl else N - Kp) * (Kb if bl else N - Kb) / N
    ax.add_patch(plt.Rectangle((cx - 0.48, cy - 0.42), 0.96, 0.84,
                               facecolor='#F6F6F6', edgecolor=GRID, lw=0.6, zorder=0))
    ax.text(cx, cy + 0.24, str(len(seeds)), ha='center', va='center',
            fontsize=15, color=(LOW if pl else HIGH))
    ax.text(cx, cy + 0.02, f'expected {exp:.1f}', ha='center', va='center',
            fontsize=6.5, color=MUTED, style='italic')
    if seeds:
        chunks = [', '.join(str(v) for v in seeds[k:k + 5]) for k in range(0, len(seeds), 5)]
        lbl = 'seeds ' + ('\n'.join(chunks))
    else:
        lbl = '--'
    ax.text(cx, cy - 0.24, lbl, ha='center', va='center', fontsize=6.2,
            color=MUTED, linespacing=1.5)
ax.text(0.5, 2.10, 'low bandwidth', ha='center', va='bottom', fontsize=7.5, color=INK)
ax.text(1.5, 2.10, 'high bandwidth', ha='center', va='bottom', fontsize=7.5, color=INK)
ax.text(-0.06, 0.5, 'low power', ha='right', va='center', fontsize=7.5,
        color=LOW, rotation=90)
ax.text(-0.06, 1.5, 'high power', ha='right', va='center', fontsize=7.5,
        color=HIGH, rotation=90)
ax.text(1.0, -0.20, 'observed against expected if the two splits were independent\n'
        'hypergeometric $P$(overlap $\\geq$ 2) = 0.613',
        ha='center', va='top', fontsize=6.8, color=MUTED)
save(fig, 'fig3_two_lotteries')
print('done ->', os.path.abspath(OUT))
