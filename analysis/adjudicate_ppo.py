#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adjudicate_ppo.py — applies PREREGISTRATION_PPO_20260908.md to the PPO sweep.

WRITTEN BEFORE ANY SEED FINISHED. It encodes the registered rules and nothing else:
the 27 dBm threshold of SIV-A (not re-tuned), the 4 dB / no-run-inside structure test X1,
the 5 pp consequence test X2 (unevaluable if either regime is empty), the 24-29 dBm
invariance test X3, and competence Gate G at the measured random-policy RLF of 12.40 %.

It prints a verdict. It does not choose one.
"""
import glob, os, sys
import numpy as np, pandas as pd

RUNS = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/ppo/runs'
THRESH   = 27.0     # SIV-A basin threshold, not re-tuned
GAP_MIN  = 4.0      # X1: minimum empty interval, the W1 bound reused
LO, HI   = 25.0, 28.0   # X1: the pre-registered interval no run may occupy
PP_MIN   = 5.0      # X2: minimum reliability difference in percentage points
GATE_RLF = 12.40    # Gate G: the uniform-random policy's sustained RLF, measured in SIII

files = sorted(glob.glob(os.path.join(RUNS, 'ppo_s*_metrics.csv')))
if not files:
    sys.exit(f'no metrics files in {RUNS}')
D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values('seed')
n = len(D)
print(f'PPO arm: {n} run(s) from {RUNS}\n')
print(D[['seed', 'return_mean', 'cvar25', 'rlf_rate', 'pt_dbm', 'rb', 'ho_per_ep', 'wall_s']]
      .to_string(index=False, float_format=lambda v: f'{v:.4f}'))

print('\n' + '=' * 70)
med_rlf = float(D.rlf_rate.median())
gate = med_rlf < GATE_RLF
print(f'GATE G  median sustained RLF {med_rlf:.2f}% vs random policy {GATE_RLF}%  '
      f'-> {"PASS" if gate else "FAIL"}')
if not gate:
    print('  Per the pre-registration, a failed gate at 1500 episodes permits exactly one')
    print('  fallback: the same twenty seeds at 6000 episodes. If it fails again the arm is')
    print('  reported as UNINFORMATIVE. It is not evidence of unimodality.')

print('=' * 70)
pt = np.sort(D.pt_dbm.values)
low = pt[pt < THRESH]; high = pt[pt >= THRESH]
inside = int(((D.pt_dbm > LO) & (D.pt_dbm < HI)).sum())
if len(low) and len(high):
    gap = float(high.min() - low.max())
else:
    gap = float('nan')
adj = float(np.max(np.diff(pt))) if len(pt) > 1 else float('nan')

print(f'  run means, sorted (dBm): {np.array2string(pt, precision=3, max_line_width=100)}')
print(f'  low regime  n={len(low)}   high regime n={len(high)}')
print(f'  empty interval across {THRESH} dBm : {gap:.3f} dB' if gap == gap else
      '  empty interval: undefined (one regime is empty)')
print(f'  largest adjacent gap anywhere      : {adj:.3f} dB')
print(f'  runs with mean inside {LO}-{HI} dBm : {inside}')

x1 = (gap == gap) and gap >= GAP_MIN and inside == 0
print(f'\nX1 (structure: gap >= {GAP_MIN} dB AND no run in {LO}-{HI})  -> '
      f'{"HELD" if x1 else "FAILED"}')

if len(low) == 0 or len(high) == 0:
    print(f'X2 (consequence: >= {PP_MIN} pp)                    -> UNEVALUABLE (one regime empty)')
else:
    rl = float(D[D.pt_dbm < THRESH].rlf_rate.mean())
    rh = float(D[D.pt_dbm >= THRESH].rlf_rate.mean())
    d = rl - rh
    print(f'X2 (consequence: >= {PP_MIN} pp)                    -> '
          f'{"HELD" if d >= PP_MIN else "FAILED"}  ({rl:.2f}% low vs {rh:.2f}% high = {d:+.2f} pp)')

cuts = np.arange(24.0, 29.01, 0.5)
labels = {c: tuple((D.pt_dbm.values < c).astype(int)) for c in cuts}
x3 = len(set(labels.values())) == 1
print(f'X3 (invariance over 24-29 dBm cuts)              -> {"HELD" if x3 else "FAILED"}')

print('\n' + '=' * 70)
if not gate:
    print('VERDICT: gate failed — the arm says nothing about bimodality either way.')
elif x1:
    print('VERDICT: the structure SURVIVES a change of algorithm family.')
    print('  Reading fixed in advance: bimodality is a property of the environment and')
    print('  objective, not of value-based learning. SVIII narrows accordingly.')
else:
    print('VERDICT: the structure DOES NOT reproduce under PPO.')
    print('  Reading fixed in advance: this is a LARGER and DIFFERENT result than the')
    print('  present paper, and the paper must be reframed around it rather than')
    print('  reporting it as an aside. See SS6 of the pre-registration.')
print('=' * 70)
