"""Fig. 5 - the empty interval is a property of incremental ADDRESSING.

Three actuators over the identical reachable power set {23..30} dBm, differing only in how many
steps it takes to cross it. Each row shows where its twenty runs converged. inc7 crosses in three
steps with more than twice the branch width and leaves the interval empty; direct8 fills it.
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIDE = '--wide' in sys.argv
SC = 1.55 if WIDE else 1.0
D = '/mnt/user-data/uploads/startfromscratch/PAPER3/figdata/fig5_actuators.csv'
OUT = '/mnt/user-data/uploads/startfromscratch/PAPER3/figures'
BLUE, ORANGE, PURPLE, GREY = '#1f6fb2', '#cf5f1f', '#7a3fa0', '#5a5a5a'
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
    'font.size':7.2*SC,'axes.labelsize':7.2*SC,'axes.titlesize':7.6*SC,
    'xtick.labelsize':6.6*SC,'ytick.labelsize':6.6*SC,
    'axes.edgecolor':'#bbbbbb','axes.linewidth':0.6,'xtick.color':GREY,'ytick.color':GREY,
    'text.color':GREY,'axes.labelcolor':GREY,'figure.dpi':200})

df = pd.read_csv(D)
spec = [('inc3', 'inc3 · 3 actions · 7 steps to cross', ORANGE),
        ('inc7', 'inc7 · 7 actions · 3 steps', BLUE),
        ('direct8', 'direct8 · 8 actions · 1 step', PURPLE)]
fig, axes = plt.subplots(3, 1, figsize=(7.16 * SC, 2.6 * SC), sharex=True)
for ax, (arm, label, c) in zip(axes, spec):
    t = df[df.arm == arm]
    ax.axvspan(25, 28, color='#000000', alpha=0.055, zorder=0, lw=0)
    committed = (t.mean_pt > 25) & (t.mean_pt < 28) & (t.frac > 0.5)
    ax.scatter(t.mean_pt[~committed], np.zeros((~committed).sum()), s=34 * SC,
               color=c, alpha=0.55, linewidths=0, zorder=2)
    ax.scatter(t.mean_pt[committed], np.zeros(committed.sum()), s=58 * SC,
               color=c, zorder=3, linewidths=0.8, edgecolors='white')
    ax.set_xlim(22.6, 30.4); ax.set_ylim(-1, 1); ax.set_yticks([])
    ax.text(0.005, 0.90, label, transform=ax.transAxes, ha='left', va='top',
            fontsize=6.6 * SC, color=c)
    n = int(committed.sum())
    ax.text(0.995, 0.90, '%d committed interior' % n, transform=ax.transAxes,
            ha='right', va='top', fontsize=6.3 * SC,
            color=(c if n else GREY), weight=('bold' if n else 'normal'))
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
axes[0].set_title('mean transmit power of each of 20 runs, by power-actuator parameterisation',
                  loc='left', pad=6)
axes[0].text(26.5, -0.55, 'empty interval 25–28 dBm', ha='center', va='center',
             fontsize=6.2 * SC, color=GREY)
axes[2].set_xlabel('mean transmit power over the held-out set (dBm)')
fig.tight_layout(pad=0.5)
suf = '_wide.png' if WIDE else '.pdf'
fig.savefig(f'{OUT}/fig5_actuators{suf}', bbox_inches='tight')
if not WIDE:
    fig.savefig(f'{OUT}/fig5_actuators.png', bbox_inches='tight', dpi=200)
print('wrote fig5_actuators' + suf)
for arm, _, _ in spec:
    t = df[df.arm == arm]
    c = ((t.mean_pt > 25) & (t.mean_pt < 28) & (t.frac > 0.5)).sum()
    print('  %-8s committed %d  span %.3f-%.3f' % (arm, c, t.mean_pt.min(), t.mean_pt.max()))
