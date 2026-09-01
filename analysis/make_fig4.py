"""Fig. 4 - the one apparently-interior run is a within-run mixture, not an interior policy.

Panel (a): per-episode mean transmit power for the single run whose RUN mean falls inside
the empty interval (sqrt seed 16), against one committed low-rail and one committed
high-rail run from the same arm. Panel (b): for every run in the power-manipulation
corpus, the fraction of its episodes that fall inside the interval.
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIDE = '--wide' in sys.argv
SC = 1.55 if WIDE else 1.0
D = '/mnt/user-data/uploads/startfromscratch/PAPER3/figdata'
OUT = '/mnt/user-data/uploads/startfromscratch/PAPER3/figures'

BLUE, ORANGE, GREY = '#1f6fb2', '#cf5f1f', '#5a5a5a'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 7.2 * SC, 'axes.labelsize': 7.2 * SC, 'axes.titlesize': 7.6 * SC,
    'xtick.labelsize': 6.6 * SC, 'ytick.labelsize': 6.6 * SC,
    'axes.edgecolor': '#bbbbbb', 'axes.linewidth': 0.6,
    'xtick.color': GREY, 'ytick.color': GREY, 'text.color': GREY,
    'axes.labelcolor': GREY, 'figure.dpi': 200,
})

ep = pd.read_csv(f'{D}/fig4a_episodes.csv')
runs = pd.read_csv(f'{D}/fig4b_runs.csv')

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(7.16 * SC, 2.75 * SC))
gs = gridspec.GridSpec(3, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.30)

# ---- panel (a): one row per run, same x-axis --------------------------------
bins = np.arange(23.0, 30.25, 0.25)
spec = [('s1_low',   'seed 1',  'low rail', ORANGE),
        ('s16',      'seed 16', 'mean 26.84 dBm', '#7a3fa0'),
        ('s0_high',  'seed 0',  'high rail', BLUE)]
axes = []
for i, (lab, name, note, c) in enumerate(spec):
    ax = fig.add_subplot(gs[i, 0])
    axes.append(ax)
    v = ep.loc[ep.run == lab, 'mean_pt'].to_numpy()
    ax.axvspan(25, 28, color='#000000', alpha=0.055, zorder=0, lw=0)
    ax.hist(v, bins=bins, color=c, histtype='stepfilled', edgecolor='none', zorder=2)
    ax.set_xlim(23, 30.2)
    ax.set_ylim(0, np.histogram(v, bins=bins)[0].max() * 1.42)
    ax.set_yticks([])
    ax.text(0.20, 0.88, '%s \u00b7 %s' % (name, note), transform=ax.transAxes,
            ha='left', va='top', fontsize=6.3 * SC, color=c)
    n_in = int(((v > 25) & (v < 28)).sum())
    ax.text(0.995, 0.88, '%d / 500 inside' % n_in, transform=ax.transAxes,
            ha='right', va='top', fontsize=6.0 * SC, color=GREY)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('mean transmit power over one held-out episode (dBm)')
axes[0].set_title('(a) three runs from the $\\sqrt{u}$ arm, episode by episode',
                  loc='left', pad=5)
axes[0].text(26.5, axes[0].get_ylim()[1] * 0.42, 'empty interval\n25\u201328 dBm',
             ha='center', va='center', fontsize=6.2 * SC, color=GREY, linespacing=1.35)

# ---- panel (b): every run in the corpus -------------------------------------
ax2 = fig.add_subplot(gs[:, 1])
y = runs.frac_ep_25_28.to_numpy() * 100
order = np.argsort(y)
mark = runs.iloc[order].reset_index(drop=True)
ys, x = y[order], np.arange(len(y))
is16 = ((mark.arm == 'sqrt') & (mark.seed == 16)).to_numpy()
ax2.scatter(x[~is16], ys[~is16], s=7 * SC, color=GREY, alpha=0.55, linewidths=0, zorder=2)
ax2.scatter(x[is16], ys[is16], s=32 * SC, color='#7a3fa0', zorder=3, linewidths=0)
ax2.annotate('sqrt seed 16\n27 % of its episodes',
             xy=(float(x[is16][0]), float(ys[is16][0])),
             xytext=(-7 * SC, 1 * SC), textcoords='offset points',
             ha='right', va='bottom', fontsize=6.2 * SC, color='#7a3fa0',
             linespacing=1.35)
ax2.axhline(50, color='#bbbbbb', lw=0.6, ls=(0, (4, 3)), zorder=1)
ax2.text(2, 52, 'a committed interior policy\nwould sit above this line',
         fontsize=6.0 * SC, color=GREY, va='bottom', linespacing=1.35)
ax2.set_ylim(-3, 100)
ax2.set_xlabel('the 120 runs, sorted')
ax2.set_ylabel('% of episodes inside 25–28 dBm')
ax2.set_title('(b) no run holds an interior power', loc='left', pad=5)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)


suffix = '_wide.png' if WIDE else '.pdf'
fig.savefig(f'{OUT}/fig4_interior_mixture{suffix}', bbox_inches='tight')
if not WIDE:
    fig.savefig(f'{OUT}/fig4_interior_mixture.png', bbox_inches='tight', dpi=200)
print('wrote fig4_interior_mixture' + suffix)
print('  runs plotted        :', len(y))
print('  max %% of episodes   : %.1f  (next highest %.1f)' % (ys[-1], ys[-2]))
print('  runs above 50%%      :', int((ys > 50).sum()))
