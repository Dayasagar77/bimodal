#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wpwr_override_proof.py -- prove the W_PWR override reaches the reward, exactly.

Not a statistical check. Runs the SAME env with the SAME reset seed and the SAME fixed action
sequence under two values of W_PWR, and asserts that at every single step

    r(w2) - r(w1)  ==  -(w2 - w1) * (pt - PT_MIN) / (PT_MAX - PT_MIN)

to machine precision. If the override were a no-op the difference would be identically zero;
if it leaked into anything besides the power term the identity would fail. It also asserts the
trajectory is otherwise untouched (same pt, same sinr, same termination), which is what makes
this a clean single-variable manipulation rather than a different experiment.

Run each arm in its own process, because paper2_env reads the override once at import.
"""
import os, subprocess, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', 'Paper2', 'code'))

CHILD = r'''
import os, sys, json
import numpy as np
sys.path.insert(0, r"%s"); os.chdir(r"%s")
import pandas as pd, paper2_env as PE

pool = pd.read_csv('dair_geometry_pool.csv')
env = PE.MultiRSUBlockageEnv(pool, seed=0, split='train')
obs, info = env.reset(seed=12345)
rng = np.random.default_rng(999)          # action stream fixed and identical in both arms
rows = []
for t in range(60):
    a = [int(rng.integers(2)), int(rng.integers(3)), int(rng.integers(3))]
    obs, r, term, trunc, i2 = env.step(a)
    rows.append(dict(t=t, r=float(r), pt=float(i2['pt']), rb=float(i2['rb']),
                     sinr=float(i2['sinr']), rlf=bool(i2['rlf']), term=bool(term)))
    if term or trunc:
        break
print("JSONSTART" + json.dumps(dict(
    w_pwr=float(PE.W_PWR), pt_min=float(PE.PT_MIN), pt_max=float(PE.PT_MAX),
    w_rel=float(PE.W_REL), rlf_pen=float(PE.RLF_PEN), w_spec=float(PE.W_SPEC),
    w_ho=float(PE.W_HO), rows=rows)))
''' % (CODE, CODE)


def arm(w):
    env = dict(os.environ)
    if w is None:
        env.pop('P2_W_PWR', None)
    else:
        env['P2_W_PWR'] = repr(float(w))
    out = subprocess.run([sys.executable, '-c', CHILD], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        sys.exit('child failed:\n' + out.stdout[-3000:] + out.stderr[-3000:])
    return json.loads(out.stdout.split('JSONSTART', 1)[1])


print('=' * 70)
print('W_PWR OVERRIDE PROOF')
print('=' * 70)

base = arm(None)
print(f'  [1] P2_W_PWR unset          -> W_PWR = {base["w_pwr"]}')
assert base['w_pwr'] == 0.15, 'DEFAULT CHANGED -- every stored result would be invalidated'
print('      OK: default is the published 0.15, unchanged')

for w in (0.05, 0.45):
    a = arm(w)
    print(f'  [2] P2_W_PWR={w!r:<6}          -> W_PWR = {a["w_pwr"]}')
    assert abs(a['w_pwr'] - w) < 1e-12, 'override did not take'

    other = {k: a[k] for k in ('w_rel', 'rlf_pen', 'w_spec', 'w_ho')}
    ref = {k: base[k] for k in ('w_rel', 'rlf_pen', 'w_spec', 'w_ho')}
    assert other == ref, f'OTHER WEIGHTS MOVED: {other} vs {ref}'
    print(f'      OK: W_REL/RLF_PEN/W_SPEC/W_HO untouched {ref}')

    assert len(a['rows']) == len(base['rows']), 'trajectory length differs'
    span = base['pt_max'] - base['pt_min']
    worst = 0.0
    for rb, ra in zip(base['rows'], a['rows']):
        assert rb['pt'] == ra['pt'] and rb['sinr'] == ra['sinr'] and rb['rlf'] == ra['rlf'], \
            't=%d: trajectory diverged -- the override touched more than the reward' % rb['t']
        expect = -(w - base['w_pwr']) * (rb['pt'] - base['pt_min']) / span
        worst = max(worst, abs((ra['r'] - rb['r']) - expect))
    print(f'      OK: trajectory identical over {len(a["rows"])} steps '
          f'(same pt, sinr, rlf at every step)')
    print(f'      OK: reward delta matches -(dW)*(pt-PT_MIN)/(PT_MAX-PT_MIN) exactly; '
          f'worst residual {worst:.3e}')
    assert worst < 1e-12, 'reward delta does not match the power term alone'
    nz = sum(1 for rb, ra in zip(base['rows'], a['rows']) if ra['r'] != rb['r'])
    print(f'      OK: {nz}/{len(a["rows"])} steps actually changed reward (a no-op would be 0)')
    assert nz > 0, 'THE OVERRIDE IS A NO-OP'

print('=' * 70)
print('PROVEN: P2_W_PWR changes the reward, changes ONLY the power term, and leaves the')
print('        trajectory and all other weights untouched. Default unset == 0.15.')
print('=' * 70)
