#!/usr/bin/env python3
"""Prove the power-penalty FORM override behaves exactly as designed.

Two things must hold, and both are checkable without any training:
  1. The RAILS are priced identically under every form. At Pt = PT_MIN and Pt = PT_MAX
     the reward must be bit-identical across linear/quad/sqrt, because f(0)=0, f(1)=1.
     If this fails the experiment is confounded: an outcome change could be the rails
     being repriced rather than the interior becoming reachable.
  2. The INTERIOR is repriced in the predicted direction: quad cheaper than linear,
     sqrt dearer, at every intermediate power.
"""
import json, os, subprocess, sys

CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'Paper2', 'code'))
CHILD = r'''
import os, sys, json
import numpy as np
sys.path.insert(0, r"%s"); os.chdir(r"%s")
import pandas as pd, paper2_env as PE
pool = pd.read_csv('dair_geometry_pool.csv')
env = PE.MultiRSUBlockageEnv(pool, seed=0, split='train')
env.reset(seed=4242)
out = []
for pt in [PE.PT_MIN + k*(PE.PT_MAX-PE.PT_MIN)/8.0 for k in range(9)]:
    u = (pt - PE.PT_MIN)/(PE.PT_MAX - PE.PT_MIN)
    out.append(dict(pt=pt, u=u, cost=PE._pwr_cost(u), pen=PE.W_PWR*PE._pwr_cost(u)))
print("JSON"+json.dumps(dict(form=PE._PWR_FORM, w=PE.W_PWR, rows=out)))
''' % (CODE, CODE)

def arm(form):
    e = dict(os.environ)
    if form is None: e.pop('P2_PWR_FORM', None)
    else: e['P2_PWR_FORM'] = form
    r = subprocess.run([sys.executable, '-c', CHILD], capture_output=True, text=True, env=e)
    if r.returncode: sys.exit(r.stdout[-2000:] + r.stderr[-2000:])
    return json.loads(r.stdout.split('JSON', 1)[1])

print('=' * 74)
print('POWER-PENALTY FORM: EXACTNESS PROOF')
print('=' * 74)
base = arm(None)
print('  P2_PWR_FORM unset -> %r  (default must stay linear)' % base['form'])
assert base['form'] == 'linear', 'DEFAULT CHANGED - every stored result invalidated'

arms = {f: arm(f) for f in ('linear', 'quad', 'sqrt')}
print()
print('  %-8s %s' % ('Pt dBm', '  '.join('%-8s' % f for f in ('linear','quad','sqrt'))))
for i in range(9):
    row = '  %-8.2f' % arms['linear']['rows'][i]['pt']
    row += '  '.join('%-8.5f' % arms[f]['rows'][i]['pen'] for f in ('linear','quad','sqrt'))
    print(row)

print()
for i, lbl in ((0, 'FLOOR  (Pt = PT_MIN)'), (8, 'CEILING(Pt = PT_MAX)')):
    v = [arms[f]['rows'][i]['pen'] for f in ('linear','quad','sqrt')]
    same = max(v) - min(v) < 1e-15
    print('  %s  %s   penalties %s' % ('OK ' if same else 'FAIL', lbl,
          ' '.join('%.6f' % x for x in v)))
    assert same, 'RAILS ARE NOT PRICED IDENTICALLY - the experiment would be confounded'

ok_q = all(arms['quad']['rows'][i]['pen'] < arms['linear']['rows'][i]['pen'] for i in range(1,8))
ok_s = all(arms['sqrt']['rows'][i]['pen'] > arms['linear']['rows'][i]['pen'] for i in range(1,8))
print('  %s  quad prices the interior CHEAPER than linear at all 7 interior points' % ('OK ' if ok_q else 'FAIL'))
print('  %s  sqrt prices the interior DEARER  than linear at all 7 interior points' % ('OK ' if ok_s else 'FAIL'))
assert ok_q and ok_s
print()
print('  PROVEN: the rails are identical under all three forms, so any change in')
print('  outcome is attributable to the interior alone.')
print('=' * 74)
