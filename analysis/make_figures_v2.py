#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the basin paper, revised 8 September 2026 against the supervisor's review.

Changes from make_figures.py / make_fig4.py / make_fig5.py, each answering one review point:
  Fig 1  a marginal distribution panel (histogram + ECDF) of mean transmit power, so the
         paper's central claim about distribution SHAPE is shown directly rather than
         inferred from a strip of points.
  Fig 2  per-run points kept, plus regime means with 95% bootstrap confidence intervals
         and explicit n, on the log reliability axis.
  Fig 4  committed interior policies and rail mixtures given different marks, so the
         distinction the paper turns on is visible without reading the caption.
  Fig 5  the three actuators labelled with their addressing and step count on the axis.
  Fig 6  NEW: a schematic of inc3 / inc7 / direct8 action representation.
Every panel carries units, a sample count and, where a statistic is drawn, its definition.

Reads stored results only. Palette and marker-shape redundancy unchanged from the original.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

HIGH, LOW, MIX = '#1668B8', '#C4561A', '#6A3D9A'
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

D = pd.read_csv(os.path.join(HERE, 'wpwr_adjudication_runs.csv'))
D['low'] = D.pt < THRESH
assert len(D) == 80, f'expected the 80-run weight sweep, got {len(D)}'


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    # a second, wider render for the Word build, where the figure spans a 6.5in page
    plt.close(fig)
    print(f'  wrote {name}.pdf / .png')


def boot_ci(x, n=20000, seed=0):
    """95% percentile bootstrap CI of the mean."""
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = r.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ============================================================ Fig 1 (two-column)
fig, (axm, axd) = plt.subplots(
    2, 1, figsize=(7.16, 3.5), sharex=True,
    gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.12))

arms = sorted(D.w.unique())
axm.axvspan(25, 28, color='#F2F2F2', zorder=0)
axm.axvline(THRESH, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
rng = np.random.default_rng(7)
for i, w in enumerate(arms):
    a = D[D.w == w]
    for mask, col, mk, ms in ((a.low, LOW, '^', 16), (~a.low, HIGH, 'o', 15)):
        s = a[mask]
        axm.scatter(s.pt, np.full(len(s), i) + rng.uniform(-.13, .13, len(s)),
                    s=ms, c=col, marker=mk, linewidths=0.5, edgecolors='white', zorder=3)
    axm.text(21.4, i, f'{len(a[a.low])}/20', ha='right', va='center',
             fontsize=7.5, color=MUTED)
axm.set_yticks(range(len(arms)))
axm.set_yticklabels([('0.15\n(reported)' if w == 0.15 else f'{w:g}') for w in arms])
axm.set_ylim(-0.6, len(arms) - 0.4)
axm.set_xlim(20.4, 30.9)
axm.set_ylabel('reward transmit-power\nweight $W_{\\mathrm{pwr}}$')
axm.text(26.5, len(arms) - 0.42, 'no run, at any weight', ha='center', va='bottom',
         fontsize=7.5, color=MUTED, style='italic')
axm.text(21.4, len(arms) - 0.45, 'low / 20', ha='right', va='bottom', fontsize=7, color=MUTED)
axm.grid(axis='x', color=GRID, lw=0.5, zorder=0)
axm.set_axisbelow(True)
for sp in ('top', 'right', 'left'):
    axm.spines[sp].set_visible(False)
axm.legend(handles=[Line2D([], [], ls='', marker='^', color=LOW, ms=4,
                           label='low-power regime ($\\bar{P}_t < 27$ dBm)'),
                    Line2D([], [], ls='', marker='o', color=HIGH, ms=4,
                           label='high-power regime ($\\bar{P}_t \\geq 27$ dBm)')],
           loc='center left', ncol=1, frameon=False, handletextpad=0.3,
           bbox_to_anchor=(0.42, 0.42))

# --- the distribution panel: what the paper is actually claiming ---------------
bins = np.arange(20.5, 31.0, 0.25)
axd.hist(D.pt, bins=bins, color='#BDBDBD', edgecolor='white', linewidth=0.3, zorder=2)
axd.axvspan(25, 28, color='#F2F2F2', zorder=0)
axd.set_ylabel('runs\nper 0.25 dB')
axd.set_xlabel('mean transmit power over the 500 held-out episodes (dBm)')
axd.grid(axis='x', color=GRID, lw=0.5, zorder=0)
axd.set_axisbelow(True)
for sp in ('top', 'right'):
    axd.spines[sp].set_visible(False)

axe = axd.twinx()
xs = np.sort(D.pt.values)
axe.step(np.concatenate([[20.4], xs, [30.9]]),
         np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs), [1]]),
         where='post', color=INK, lw=0.9, zorder=4)
axe.set_ylim(-0.02, 1.02)
axe.set_ylabel('ECDF', color=INK)
axe.tick_params(axis='y', colors=MUTED, labelsize=7.5)
for sp in ('top',):
    axe.spines[sp].set_visible(False)
gap_lo, gap_hi = D[D.low].pt.max(), D[~D.low].pt.min()
axd.annotate('', xy=(gap_lo, axd.get_ylim()[1] * 0.72), xytext=(gap_hi, axd.get_ylim()[1] * 0.72),
             arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.7))
axd.text((gap_lo + gap_hi) / 2, axd.get_ylim()[1] * 0.78,
         f'empty interval {gap_hi - gap_lo:.2f} dB\n(0 of 80 runs)',
         ha='center', va='bottom', fontsize=7, color=MUTED, style='italic')
save(fig, 'fig1_basins_by_weight')

# ============================================================ Fig 2 (one-column)
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.axvspan(25, 28, color='#F2F2F2', zorder=0)
for mask, col, mk, lab in ((D.low, LOW, '^', 'low-power regime'),
                           (~D.low, HIGH, 'o', 'high-power regime')):
    s = D[mask]
    ax.scatter(s.pt, s.rlf, s=13, c=col, marker=mk, linewidths=0.4,
               edgecolors='white', zorder=3, label=f'{lab} ($n$ = {len(s)})', alpha=0.85)
ax.set_yscale('log')
ax.set_xlabel('mean transmit power (dBm)')
ax.set_ylabel('sustained radio-link failure (%),\nlog scale')
ax.set_xlim(22.4, 30.8)
ax.grid(color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

# regime means with 95% bootstrap CIs, drawn on top of the per-run points
for mask, col, x0 in ((D.low, LOW, 23.30), (~D.low, HIGH, 29.60)):
    v = D[mask].rlf.values
    m = v.mean(); lo, hi = boot_ci(v)
    ax.errorbar([x0], [m], yerr=[[m - lo], [hi - m]], fmt='_', color=col,
                ms=14, elinewidth=1.4, capsize=3, capthick=1.2, zorder=5)
    ax.annotate(f'{m:.2f}%\n[{lo:.2f}, {hi:.2f}]', (x0, m),
                xytext=(7 if col == LOW else -7, 0), textcoords='offset points',
                fontsize=6.8, color=col, ha='left' if col == LOW else 'right', va='center')
ax.legend(loc='lower left', frameon=False, handletextpad=0.3, borderpad=0.1, fontsize=6.8)
ax.text(0.99, 0.98, 'bars: regime mean, 95% bootstrap CI\n(20,000 resamples)',
        transform=ax.transAxes, ha='right', va='top', fontsize=6.2, color=MUTED, style='italic')
save(fig, 'fig2_rlf_vs_power')
print('figures 1-2 done')


# ============================================================ Fig 4 (two-column)
# Review point: make the distinction between "mean inside" and "committed interior"
# visually explicit — it is the distinction the whole of SIV-G and SIV-H turns on.
EP = pd.read_csv(os.path.join(HERE, 'fig4a_episodes.csv'))
RB = pd.read_csv(os.path.join(HERE, 'fig4b_runs.csv'))
assert len(RB) == 120, f'expected the 120 power-objective runs, got {len(RB)}'

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.5),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.28))

order = [('s1', 'seed 1 · committed low rail', LOW),
         ('s16', 'seed 16 · mean 26.84 dBm', MIX),
         ('s0', 'seed 0 · committed high rail', HIGH)]
present = [o for o in order if o[0] in set(EP.run)]
bins = np.arange(23.0, 30.25, 0.25)
for k, (run, lab, col) in enumerate(present):
    v = EP[EP.run == run].mean_pt.values
    inside = ((v > 25) & (v < 28)).mean()
    off = k
    h, _ = np.histogram(v, bins=bins)
    a1.bar(bins[:-1], h / h.max() * 0.8, bottom=off, width=0.23, align='edge',
           color=col, linewidth=0, zorder=3)
    a1.axhline(off, color=GRID, lw=0.5, zorder=1)
    a1.text(23.05, off + 0.62, lab, fontsize=6.8, color=col, va='center')
    a1.text(30.15, off + 0.62, f'{inside*100:.0f}% of 500 episodes inside',
            fontsize=6.4, color=MUTED, ha='right', va='center')
a1.axvspan(25, 28, color='#F2F2F2', zorder=0)
a1.set_ylim(-0.15, len(present) + 0.05)
a1.set_yticks([]); a1.set_xlim(22.9, 30.3)
a1.set_xlabel('mean transmit power of one held-out episode (dBm)')
a1.set_title('(a) per-episode distribution, three runs of the $\\sqrt{u}$ arm', loc='left', pad=5)
for sp in ('top', 'right', 'left'):
    a1.spines[sp].set_visible(False)
a1.text(26.5, len(present) - 0.04, 'empty interval 25–28 dBm', ha='center', va='bottom',
        fontsize=6.4, color=MUTED, style='italic')

# --- (b): mean-inside vs committed, now with THREE visually distinct classes ---
RB = RB.sort_values('frac_ep_25_28').reset_index(drop=True)
RB['mean_inside'] = (RB.run_mean_pt > 25) & (RB.run_mean_pt < 28)
RB['committed'] = RB.mean_inside & (RB.frac_ep_25_28 > 0.5)
cls = np.where(RB.committed, 'committed', np.where(RB.mean_inside, 'mixture', 'rail'))
style = {'rail': (MUTED, 'o', 8, 0.45, 'rail policy'),
         'mixture': (MIX, 'D', 34, 1.0, 'mean inside 25–28 dBm,\nbut a rail mixture'),
         'committed': ('#1B7837', '*', 90, 1.0, 'committed interior policy')}
for k in ('rail', 'mixture', 'committed'):
    m = cls == k
    c, mk, s, al, lab = style[k]
    a2.scatter(np.arange(len(RB))[m], RB.frac_ep_25_28[m] * 100, s=s, c=c, marker=mk,
               alpha=al, linewidths=0.4, edgecolors='white', zorder=3,
               label=f'{lab} ($n$ = {int(m.sum())})')
a2.axhline(50, color=MIX, lw=0.8, ls=(0, (4, 3)), zorder=2)
a2.text(2, 53, 'committed threshold: >50% of episodes inside',
        fontsize=6.2, color=MIX, va='bottom')
a2.set_ylim(-4, 104)
a2.set_xlabel('the 120 runs in which the power objective was manipulated, sorted')
a2.set_ylabel('% of held-out episodes\nwith mean inside 25–28 dBm')
a2.set_title('(b) no run holds an interior policy', loc='left', pad=5)
a2.grid(axis='y', color=GRID, lw=0.5, zorder=0)
a2.set_axisbelow(True)
for sp in ('top', 'right'):
    a2.spines[sp].set_visible(False)
a2.legend(loc='upper left', frameon=False, fontsize=6.2, handletextpad=0.3,
          borderpad=0.1, labelspacing=0.6)
save(fig, 'fig4_interior_mixture')

# ============================================================ Fig 5 (two-column)
AC = pd.read_csv(os.path.join(HERE, 'fig5_actuators.csv'))
assert len(AC) == 60, f'expected the 60-run actuator sweep, got {len(AC)}'
spec = [('inc3',    'inc3',    'relative',  3, 7, LOW),
        ('inc7',    'inc7',    'relative',  7, 3, HIGH),
        ('direct8', 'direct8', 'ABSOLUTE',  8, 1, MIX)]
fig, axes = plt.subplots(3, 1, figsize=(7.16, 2.9), sharex=True,
                         gridspec_kw=dict(hspace=0.22))
for ax, (arm, name, addr, nact, steps, c) in zip(axes, spec):
    s = AC[AC.arm == arm]
    assert len(s) == 20, f'{arm}: {len(s)} runs'
    ax.axvspan(25, 28, color='#F2F2F2', zorder=0)
    committed = (s.mean_pt > 25) & (s.mean_pt < 28) & (s.frac > 0.5)
    ax.scatter(s.mean_pt[~committed], np.zeros((~committed).sum()), s=26,
               color=c, alpha=0.5, linewidths=0, zorder=2)
    ax.scatter(s.mean_pt[committed], np.zeros(committed.sum()), s=95, color='#1B7837',
               marker='*', zorder=3, linewidths=0.5, edgecolors='white')
    ax.set_xlim(22.6, 30.4); ax.set_ylim(-1, 1); ax.set_yticks([])
    ax.set_ylabel(f'{name}\n{addr} addressing\n{nact} actions · {steps} step'
                  + ('s' if steps > 1 else ''),
                  rotation=0, ha='right', va='center', fontsize=6.6,
                  color=(INK if addr == 'ABSOLUTE' else MUTED))
    n = int(committed.sum())
    ax.text(0.995, 0.88, f'{n} committed interior of 20', transform=ax.transAxes,
            ha='right', va='top', fontsize=6.4,
            color=('#1B7837' if n else MUTED), weight=('bold' if n else 'normal'))
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
axes[0].set_title('mean transmit power of each of 20 runs, by power-actuator parameterisation.  '
                  'All three actuators reach the identical set {23…30} dBm.', loc='left', pad=6,
                  fontsize=7.2)
axes[1].text(26.5, -0.72, 'empty interval 25–28 dBm', ha='center', va='center',
             fontsize=6.4, color=MUTED, style='italic')
axes[2].set_xlabel('mean transmit power over the 500 held-out episodes (dBm)')
axes[2].legend(handles=[Line2D([], [], ls='', marker='*', color='#1B7837', ms=7,
                               label='committed interior policy'),
                        Line2D([], [], ls='', marker='o', color=MUTED, ms=4,
                               label='rail or rail-mixture run')],
               loc='lower left', bbox_to_anchor=(0.0, -1.35), ncol=2, frameon=False,
               fontsize=6.4, handletextpad=0.3)
save(fig, 'fig5_actuators')
print('figures 4-5 done')


# ============================================================ Fig 6 (NEW, 2-col)
# Review point: "add a small figure showing inc3 -> inc7 -> direct8 action representation
# visually". Nothing here is data — it is a diagram of the three action spaces, drawn from
# their definitions in Table VII.
LEVELS = list(range(23, 31))
START = 30                      # every episode resets at the ceiling
TARGET = 26                     # a representative interior level

fig, axes = plt.subplots(3, 1, figsize=(7.16, 3.0), sharex=True,
                         gridspec_kw=dict(hspace=0.45))
rows = [('inc3',    'relative',  [-1, 0, +1],                    LOW,
         'one action changes power by at most 1 dB'),
        ('inc7',    'relative',  [-3, -2, -1, 0, +1, +2, +3],    HIGH,
         'one action changes power by at most 3 dB'),
        ('direct8', 'absolute',  None,                            MIX,
         'one action names the level outright')]

for ax, (name, addr, deltas, col, note) in zip(axes, rows):
    ax.set_xlim(22.4, 30.6); ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([]); ax.set_xticks(LEVELS)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.axvspan(25, 28, color='#F2F2F2', zorder=0)
    ax.hlines(0, 22.6, 30.4, color=GRID, lw=0.8, zorder=1)
    ax.scatter(LEVELS, np.zeros(len(LEVELS)), s=22, c='white', edgecolors=MUTED,
               linewidths=0.8, zorder=3)
    ax.scatter([START], [0], s=40, c=INK, zorder=4)
    ax.text(START, 0.30, 'reset\nstate', ha='center', va='bottom', fontsize=6.0, color=INK)

    if deltas is None:                        # direct8: one arrow to every level
        for L in LEVELS:
            if L == START:
                continue
            ax.annotate('', xy=(L, -0.42), xytext=(START, -0.05),
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.7,
                                        connectionstyle='arc3,rad=0.16', alpha=0.85))
        steps = 1
    else:                                     # incremental: arrows to reachable neighbours
        for d in deltas:
            if d == 0:
                continue
            tgt = min(max(START + d, 23), 30)
            ax.annotate('', xy=(tgt, -0.42), xytext=(START, -0.05),
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.8,
                                        connectionstyle='arc3,rad=0.20'))
        steps = int(np.ceil((START - TARGET) / max(abs(min(deltas)), 1)))

    ax.set_ylabel(f'{name}\n{addr}', rotation=0, ha='right', va='center',
                  fontsize=7.0, color=(INK if addr == 'absolute' else MUTED))
    ax.text(22.55, 0.72, note, fontsize=6.3, color=col, ha='left', va='center')
    ax.text(22.55, -0.78,
            f'steps from the reset state to {TARGET} dBm: '
            + ('$\\mathbf{1}$' if steps == 1 else str(steps)),
            fontsize=6.3, color=MUTED, ha='left', va='center')

axes[0].set_title('the same reachable power set {23…30} dBm, addressed three ways.  '
                  'Arrows show where one action can take the policy from the reset state.',
                  loc='left', pad=6, fontsize=7.2)
axes[2].set_xlabel('transmit power (dBm).  Shaded band: the 25–28 dBm interval of §IV-C.')
save(fig, 'fig6_action_representation')
print('figure 6 done')
