#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wpwr_adjudicate.py -- score the W_PWR sweep against the 30 Aug pre-registration.

Reads only stored _metrics.csv files. Recomputes everything from source; nothing is carried
over from a progress report. W1-W4 and the kill rule are quoted verbatim from
claude/DnDQN_Paper3_WPWR_Sweep_PREREGISTRATION_20260830.md.
"""
import glob, os, re, sys
import numpy as np, pandas as pd

THRESH = 27.0
WSTAR = 0.0520          # pre-computed indifference point, from basin_v2 alone
ARMS = {0.02: 'models/wpwr_w0.02', 0.05: 'models/wpwr_w0.05',
        0.15: 'models/basin_v2', 0.45: 'models/wpwr_w0.45'}

rows, mism = [], []
for w, d in ARMS.items():
    for f in sorted(glob.glob(os.path.join(d, 'model_s*_metrics.csv'))):
        s = int(re.search(r'model_s(\d+)_', os.path.basename(f)).group(1))
        r = pd.read_csv(f).iloc[0]
        if 'w_pwr' in r.index and abs(float(r['w_pwr']) - w) > 1e-12:
            mism.append((w, s, float(r['w_pwr'])))
        rows.append(dict(w=w, seed=s, pt=float(r['pt_dbm']), rlf=float(r['rlf_rate']),
                         ret=float(r['return_mean']), cvar=float(r['cvar25']),
                         steps=float(r['steps'])))
D = pd.DataFrame(rows)
D['low'] = D.pt < THRESH

print('=' * 78)
print('W_PWR SWEEP -- FINAL ADJUDICATION against the 30 Aug 2026 pre-registration')
print('=' * 78)
print(f'\nG3 GUARD: {len(D)} runs ({len(D[D.w!=0.15])} new + 20 baseline), '
      f'{len(mism)} weight mismatches -> {"ALL CLEAN" if not mism else "CONTAMINATED"}')
for w, s, got in mism:
    print(f'   arm {w} seed {s} recorded {got}')

print('\n' + '-' * 78)
print('PER-ARM TABLE   (pt_dbm and rlf_rate are physical and comparable across arms;')
print('                 return/cvar are NOT -- the arms optimise different objectives)')
print('-' * 78)
hdr = f"{'W_PWR':>6} {'n':>3} {'low':>6} {'low max':>9} {'high min':>9} {'gap dB':>8} {'25-28':>6} {'RLF low':>8} {'RLF hi':>8} {'diff pp':>8}"
print(hdr); print('-' * len(hdr))
summary = {}
for w in sorted(D.w.unique()):
    a = D[D.w == w]
    lo, hi = a[a.low], a[~a.low]
    gap = hi.pt.min() - lo.pt.max() if (len(lo) and len(hi)) else np.nan
    mid = int(((a.pt > 25) & (a.pt < 28)).sum())
    dl = lo.rlf.mean() - hi.rlf.mean() if (len(lo) and len(hi)) else np.nan
    print(f"{w:>6} {len(a):>3} {f'{len(lo)}/{len(a)}':>6} {lo.pt.max():>9.3f} "
          f"{hi.pt.min():>9.3f} {gap:>8.3f} {mid:>6} {lo.rlf.mean():>8.2f} "
          f"{hi.rlf.mean():>8.2f} {dl:>+8.2f}")
    # rank separation: every low-vs-high pair
    pairs = [(l, h) for l in lo.rlf for h in hi.rlf]
    win = sum(1 for l, h in pairs if l > h)
    summary[w] = dict(n=len(a), n_low=len(lo), gap=gap, mid=mid, dl=dl,
                      pairs=len(pairs), win=win)

print('\nrank separation (low-basin run has HIGHER RLF than high-basin run):')
for w in sorted(summary):
    s = summary[w]
    print(f'   W_PWR={w:<5} {s["win"]}/{s["pairs"]} pairs '
          f'({100.0*s["win"]/s["pairs"]:.1f}%)')

# pooled permutation test on the pooled RLF difference
rng = np.random.default_rng(20260830)
lo_all, hi_all = D[D.low].rlf.to_numpy(), D[~D.low].rlf.to_numpy()
obs = lo_all.mean() - hi_all.mean()
pool = np.concatenate([lo_all, hi_all]); n1 = len(lo_all)
cnt = sum(1 for _ in range(200000)
          if (lambda p: p[:n1].mean() - p[n1:].mean() >= obs)(rng.permutation(pool)))
print(f'\npooled across all {len(D)} runs, all four weights:')
print(f'   low  n={len(lo_all):2d}  RLF {lo_all.mean():.2f} +/- {lo_all.std(ddof=1):.2f}')
print(f'   high n={len(hi_all):2d}  RLF {hi_all.mean():.2f} +/- {hi_all.std(ddof=1):.2f}')
print(f'   difference {obs:+.2f} pp, ratio {lo_all.mean()/hi_all.mean():.1f}x')
allpairs = len(lo_all) * len(hi_all)
allwin = sum(1 for l in lo_all for h in hi_all if l > h)
print(f'   {allwin}/{allpairs} pairwise comparisons favour the high-power basin')
print(f'   one-sided permutation p = {(cnt+1)/200001:.2e} ({cnt} of 200000 reached it)')

print('\n' + '=' * 78)
print('PRE-REGISTERED PREDICTIONS')
print('=' * 78)
ok = lambda b: 'HOLDS' if b else '*** FAILS ***'

w1 = all(summary[w]['gap'] >= 4.0 and summary[w]['mid'] == 0
         for w in summary if summary[w]['n_low'] >= 3 and summary[w]['n'] - summary[w]['n_low'] >= 3)
print(f'\nW1  structure: gap >= 4 dB and ZERO runs in 25-28 dBm, in every arm with')
print(f'    >=3 runs each side.                                            {ok(w1)}')
for w in sorted(summary):
    print(f'      W_PWR={w:<5} gap {summary[w]["gap"]:.3f} dB, {summary[w]["mid"]} runs in 25-28')

seq = [summary[w]['n_low'] for w in sorted(summary)]
w2 = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
print(f'\nW2  propensity monotone in W_PWR (WEAK test; +/-2 runs declared meaningless')
print(f'    in advance).  observed {seq} for W_PWR {sorted(summary)}   {ok(w2)}')
if not w2:
    for i in range(len(seq)-1):
        if seq[i] > seq[i+1]:
            a_, b_ = sorted(summary)[i], sorted(summary)[i+1]
            print(f'      inversion: n_low({a_})={seq[i]} > n_low({b_})={seq[i+1]}, '
                  f'a difference of {seq[i]-seq[i+1]} run(s)')
    print(f'      direction across the full 22x range: n_low({sorted(summary)[0]})={seq[0]}'
          f' -> n_low({sorted(summary)[-1]})={seq[-1]}')

w3 = all(summary[w]['dl'] >= 5.0 for w in summary
         if summary[w]['n_low'] >= 3 and summary[w]['n'] - summary[w]['n_low'] >= 3)
print(f'\nW3  low-basin RLF exceeds high-basin RLF by >= 5 pp in every arm with')
print(f'    >=3 runs each side.                                            {ok(w3)}')
for w in sorted(summary):
    print(f'      W_PWR={w:<5} {summary[w]["dl"]:+.2f} pp')

n02 = summary[0.02]['n_low']
w4 = n02 >= 1
print(f'\nW4  DECISIVE: at W_PWR=0.02, where the objective prefers HIGH power by')
print(f'    +2.97, at least one run still lands in the LOW basin.          {ok(w4)}')
print(f'      {n02} of {summary[0.02]["n"]} runs converged to minimum power against the incentive')

unan = lambda w: summary[w]['n_low'] in (0, summary[w]['n'])
kill = unan(0.02) and unan(0.45) and (summary[0.02]['n_low'] == 0) != (summary[0.45]['n_low'] == 0)
print(f'\nKILL RULE  fires only if arms 0.02 and 0.45 are EACH unanimous in opposite')
print(f'           basins.  0.02 is {summary[0.02]["n_low"]}/{summary[0.02]["n"]}, '
      f'0.45 is {summary[0.45]["n_low"]}/{summary[0.45]["n"]}.        '
      f'{"*** FIRED ***" if kill else "NOT TRIGGERED"}')

print('\n' + '=' * 78)
print('NOT PRE-REGISTERED -- descriptive: which seeds moved, and against what incentive')
print('=' * 78)
base = {int(r.seed): bool(r.low) for _, r in D[D.w == 0.15].iterrows()}
print('  the objective favours HIGH below W*=0.0520 and LOW above it\n')
for w in sorted(summary):
    if w == 0.15:
        continue
    a = D[D.w == w].sort_values('seed')
    fav_low = w > WSTAR
    flips = [(int(r.seed), base[int(r.seed)], bool(r.low)) for _, r in a.iterrows()
             if base[int(r.seed)] != bool(r.low)]
    against = [s for s, b, n in flips if n != fav_low]
    withi = [s for s, b, n in flips if n == fav_low]
    print(f'  W_PWR={w:<5} objective favours {"LOW " if fav_low else "HIGH"}  '
          f'{len(flips)} of 20 seeds changed basin vs the published weight')
    print(f'{"":16}moved WITH the incentive:    {len(withi)}  {sorted(withi)}')
    print(f'{"":16}moved AGAINST the incentive: {len(against)}  {sorted(against)}')
D.sort_values(['w', 'seed']).to_csv('results/wpwr_adjudication_runs.csv', index=False)
print('\nsaved -> results/wpwr_adjudication_runs.csv')
