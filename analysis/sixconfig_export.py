"""Export and verify the six-configuration controller sweep (Sec. IV-D, Sec. VI-A).

These 120 runs were produced for the ablation study of [4] and are not part of the 180-run
record in runs/. This script writes the per-run summary and the per-episode return series that
Sec. IV-D and Sec. VI-A are computed from, and REFUSES to write anything unless the summary
reproduces the published Table IV, the per-seed basin counts and the chi-square test exactly.

Usage:  python sixconfig_export.py --src <path to models/sweep_r3> [--write]
"""
import argparse, csv, math, os, shutil, sys
import numpy as np

# NumPy is used for the means on purpose. Two cells of Table IV sit exactly on a rounding
# boundary -- the minus-prioritised-replay high-basin RLF is exactly 4.425 % and the plus-LSTM
# high-basin RLF exactly 1.575 % -- so the printed second decimal depends on summation order.
# The published table was produced with numpy's pairwise summation; a plain left-to-right
# sum(v)/len(v) lands on the other side of both ties and prints 4.42 and 1.58. The values are
# identical; only the tie-break differs. Anything else here is plain Python so that the script
# runs with no SciPy dependency.
def _mean(v):  return float(np.mean(v))
def _corr(x, y):
    mx, my = _mean(x), _mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
def _binom_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)
def _chi2_sf(x, df):
    """Upper tail of chi-square for even df: exp(-x/2) * sum_{i<df/2} (x/2)^i / i!."""
    assert df % 2 == 0
    h = x / 2.0
    return math.exp(-h) * sum(h ** i / math.factorial(i) for i in range(df // 2))

CONFIGS = [('model', 'reported'), ('no_ant', 'minus anticipation'), ('no_cvar', 'minus CVaR'),
           ('no_per', 'minus prioritised replay'), ('p1_transplant', 'single-agent transplant'),
           ('plus_lstm', 'plus LSTM')]
# Table IV as published: low/20, empty interval (dB), RLF low (%), RLF high (%), corr(RLF, power)
PUBLISHED = {'model':         (6,  5.85,  8.90,  0.96, -0.920),
             'no_ant':        (2,  6.47, 19.70, 11.81, -0.909),
             'no_cvar':       (13, 6.66,  8.38,  1.49, -0.875),
             'no_per':        (12, 5.68, 10.60,  4.43, -0.638),
             'p1_transplant': (2,  6.46, 21.90, 11.93, -0.845),
             'plus_lstm':     (4,  5.58,  9.15,  1.57, -0.919)}
PUBLISHED_SEED_COUNTS = [0, 5, 11, 4, 0, 0, 0]     # seeds with 0..6 low-basin outcomes
PUBLISHED_CHI2, PUBLISHED_P = 6.76, 0.344
THRESH = 27.0

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--write', action='store_true')
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(os.path.join(a.src, 'all_metrics.csv')))
        if r['config'] in dict(CONFIGS)]
assert len(rows) == 120, 'expected 120 runs across the six configurations, got %d' % len(rows)
for r in rows:
    r['pt_dbm'] = float(r['pt_dbm']); r['rlf_rate'] = float(r['rlf_rate'])
    r['cvar25'] = float(r['cvar25']); r['return_mean'] = float(r['return_mean'])
    r['seed'] = int(r['seed']); r['basin'] = 'low' if r['pt_dbm'] < THRESH else 'high'

fail = []
print('%-24s %-16s %-16s %-16s %-16s %s' % ('config', 'low/20', 'empty interval', 'RLF low', 'RLF high', 'corr'))
for key, label in CONFIGS:
    g = [r for r in rows if r['config'] == key]
    assert len(g) == 20, '%s has %d runs' % (key, len(g))
    lo = [r for r in g if r['basin'] == 'low']; hi = [r for r in g if r['basin'] == 'high']
    gap = min(r['pt_dbm'] for r in hi) - max(r['pt_dbm'] for r in lo)
    rl = _mean([r['rlf_rate'] for r in lo]); rh = _mean([r['rlf_rate'] for r in hi])
    cc = float(np.corrcoef([r['rlf_rate'] for r in g], [r['pt_dbm'] for r in g])[0, 1])
    p = PUBLISHED[key]
    got = (len(lo), round(gap, 2), round(rl, 2), round(rh, 2), round(cc, 3))
    ok = (got[0] == p[0] and abs(got[1]-p[1]) < 5e-3 and abs(got[2]-p[2]) < 5e-3
          and abs(got[3]-p[3]) < 5e-3 and abs(got[4]-p[4]) < 5e-4)
    if not ok: fail.append('%s: computed %s vs published %s' % (key, got, p))
    for nm, val in (('RLF low', rl), ('RLF high', rh)):
        if abs(val * 100 - round(val * 100)) > 0.4999 and abs(val * 1000 % 10 - 5) < 1e-6:
            print('    note: %s %s = %.4f sits exactly on a rounding boundary' % (label, nm, val))
    print('%-24s %-16s %-16s %-16s %-16s %s   %s'
          % (label, '%d / 20' % len(lo), '%.2f dB' % gap, '%.2f %%' % rl, '%.2f %%' % rh,
             '%.3f' % cc, 'OK' if ok else 'MISMATCH'))

# per-seed basin counts across the six configurations, and the chi-square test of Sec. IV-D
per_seed = [sum(1 for r in rows if r['seed'] == s and r['basin'] == 'low') for s in range(20)]
obs = [per_seed.count(k) for k in range(7)]
pi = sum(1 for r in rows if r['basin'] == 'low') / len(rows)
exp = [20 * _binom_pmf(k, 6, pi) for k in range(7)]
chi2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
pval = _chi2_sf(chi2, 6)
print('\nper-seed low-basin counts (0..6): %s   published %s' % (obs, PUBLISHED_SEED_COUNTS))
print('pooled low-basin rate %.3f   chi2 = %.2f on 6 df   p = %.3f   (published %.2f / %.3f)'
      % (pi, chi2, pval, PUBLISHED_CHI2, PUBLISHED_P))
if obs != PUBLISHED_SEED_COUNTS: fail.append('per-seed counts %s vs published %s' % (obs, PUBLISHED_SEED_COUNTS))
if abs(chi2 - PUBLISHED_CHI2) > 5e-3: fail.append('chi2 %.4f vs published %.2f' % (chi2, PUBLISHED_CHI2))
if abs(pval - PUBLISHED_P) > 5e-4: fail.append('p %.4f vs published %.3f' % (pval, PUBLISHED_P))

if fail:
    print('\nREFUSING TO WRITE — the export does not reproduce the paper:')
    for f in fail: print('  ' + f)
    sys.exit(1)
print('\nAll published values reproduced.')

if not a.write:
    print('(dry run; pass --write to emit the files)'); sys.exit(0)

# Everything lands in one directory, and the summary is named all_metrics.csv, so that
#     python basin_early_detect.py --dir runs/sixconfig
# reproduces Sec. VI-A with no further wiring.
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'sixconfig')
ret = out; os.makedirs(out, exist_ok=True)
cols = ['config', 'seed', 'alpha', 'lstm', 'anticipation', 'per', 'pt_dbm', 'rlf_rate',
        'cvar25', 'return_mean', 'basin']
with open(os.path.join(out, 'all_metrics.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in sorted(rows, key=lambda r: (r['config'], r['seed'])):
        w.writerow({c: r[c] for c in cols})
n = 0
for key, _ in CONFIGS:
    for s in range(20):
        src = os.path.join(a.src, '%s_s%d_returns.csv' % (key, s))
        if not os.path.exists(src): sys.exit('missing return series: ' + src)
        shutil.copyfile(src, os.path.join(ret, '%s_s%d_returns.csv' % (key, s))); n += 1
print('wrote %s/all_metrics.csv (120 rows) and %d return series' % (out, n))
