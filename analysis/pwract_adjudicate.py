#!/usr/bin/env python3
"""Score the power-actuator sweep against its pre-registration (A1-A4).

Committed interior policy, as pre-registered:
    run mean strictly in 25-28 dBm  AND  >50% of held-out episodes' own means inside 25-28.
A run whose MEAN lands inside but which alternates between rails across episodes is NOT one;
that distinction is the lesson of the SIV-G result and was fixed before these runs existed.
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', 'Paper2', 'code'))
THRESH, LO, HI, COMMIT = 27.0, 25.0, 28.0, 0.50


def arm(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'model_s*_eval.csv'))):
        s = int(re.search(r'model_s(\d+)_', os.path.basename(f)).group(1))
        v = pd.read_csv(f)
        pt = v['mean_pt'].to_numpy()
        # RLF from the per-episode eval file, exactly as every already-reported
        # basin RLF in this paper was computed. metrics.csv calls it rlf_rate.
        rlf = 100 * float(v['rlf'].mean())
        rows.append(dict(seed=s, pt=pt.mean(), rlf=rlf,
                         frac=float(((pt > LO) & (pt < HI)).mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(CODE, 'models'))
    a = ap.parse_args()
    arms = [('inc3 (published)', os.path.join(a.base, 'basin_v2')),
            ('inc7 (control)', os.path.join(a.base, 'pwract_inc7')),
            ('direct8 (treatment)', os.path.join(a.base, 'pwract_direct8'))]
    T = {}
    print('=' * 92)
    print('POWER-ACTUATOR SWEEP -- adjudication against the pre-registration')
    print('=' * 92)
    print('%-20s %4s %7s %10s %9s %9s %9s %9s'
          % ('actuator', 'n', 'low/n', 'COMMITTED', 'mean-in', 'max %ep', 'low max', 'high min'))
    print('-' * 92)
    for name, d in arms:
        if not os.path.isdir(d):
            print('%-20s  (not present)' % name); continue
        t = arm(d); T[name.split()[0]] = t
        if not len(t):
            print('%-20s  (empty)' % name); continue
        lo, hi = t[t.pt < THRESH], t[t.pt >= THRESH]
        comm = t[(t.pt > LO) & (t.pt < HI) & (t.frac > COMMIT)]
        print('%-20s %4d %7s %10d %9d %8.0f%% %9s %9s'
              % (name, len(t), '%d/%d' % (len(lo), len(t)), len(comm),
                 int(((t.pt > LO) & (t.pt < HI)).sum()), 100 * t.frac.max(),
                 ('%.3f' % lo.pt.max()) if len(lo) else '-',
                 ('%.3f' % hi.pt.min()) if len(hi) else '-'))
    print()
    if 'direct8' not in T or not len(T['direct8']):
        print('direct8 not yet complete -- nothing to adjudicate.'); return
    d8 = T['direct8']
    i7 = T.get('inc7', pd.DataFrame(columns=d8.columns))
    c8 = d8[(d8.pt > LO) & (d8.pt < HI) & (d8.frac > COMMIT)]
    c7 = i7[(i7.pt > LO) & (i7.pt < HI) & (i7.frac > COMMIT)] if len(i7) else i7

    print('=' * 92); print('PREDICTIONS'); print('=' * 92)
    a1 = len(c8) >= 1
    print('  A1 DECISIVE   %-6s direct8 committed interior policies: %d (predicted >= 1)'
          % ('HOLDS' if a1 else 'FAILS', len(c8)))
    if len(i7):
        a2 = len(c8) > len(c7)
        print('  A2 ATTRIBUTION %-5s direct8 %d vs inc7 %d committed (predicted direct8 strictly more)'
              % ('HOLDS' if a2 else 'FAILS', len(c8), len(c7)))
        if len(c8) == len(c7) and len(c8) > 0:
            print('       -> equal and non-zero: traversal speed or branch width, NOT addressing.')
    else:
        print('  A2 ATTRIBUTION       inc7 not present -- cannot adjudicate')
    if len(c8):
        lo8, hi8 = d8[d8.pt < THRESH], d8[d8.pt >= THRESH]
        rl, rh = lo8.rlf.mean(), hi8.rlf.mean()
        ok = all(min(rl, rh) < r < max(rl, rh) for r in c8.rlf)
        print('  A3 CONSEQUENCE %-5s interior RLF %s vs rails low %.2f%% / high %.2f%%'
              % ('HOLDS' if ok else 'FAILS',
                 ', '.join('%.2f%%' % r for r in c8.rlf), rl, rh))
    else:
        print('  A3 CONSEQUENCE       no committed interior policy -- not applicable')
    msgs = []
    for nm, t in T.items():
        if nm == 'inc3' or not len(t):
            continue
        lo, hi = t[t.pt < THRESH], t[t.pt >= THRESH]
        if len(lo) and lo.pt.max() > 23.6:
            msgs.append('%s low max %.3f > 23.6 (by %.3f dB)' % (nm, lo.pt.max(), lo.pt.max() - 23.6))
        if len(hi) and hi.pt.min() < 28.9:
            msgs.append('%s high min %.3f < 28.9 (by %.3f dB)' % (nm, hi.pt.min(), 28.9 - hi.pt.min()))
    if not msgs:
        print('  A4 RAILS       HOLDS rail locations unchanged in every arm')
    else:
        worst = max(float(re.search(r'by ([\d.]+) dB', m).group(1)) for m in msgs)
        print('  A4 RAILS       %s %s' % ('BREACH<0.25dB' if worst < 0.25 else 'VOIDS ARM',
                                          '; '.join(msgs)))
        print('       -> pre-registered rule: a breach under 0.25 dB is reported, not voiding.')
    print()
    if len(c8):
        print('Per-episode decomposition is REQUIRED for every committed interior policy')
        print('(the SIV-G lesson). Seeds to decompose:', list(c8.seed))


if __name__ == '__main__':
    main()
