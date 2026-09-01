#!/usr/bin/env python3
"""END-TO-END: does the form override actually reach the REWARD?

pwrform_proof.py checked _pwr_cost() in isolation. That is not the same as checking the
reward uses it -- exactly the class of gap that made an earlier sweep vacuous. This runs
the same env, same reset seed, same fixed action sequence under two forms and asserts:

    r_quad(t) - r_linear(t)  ==  -W_PWR * (u_t^2 - u_t)     at EVERY step

and that the trajectory (pt, sinr, rlf) is untouched.
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
env.reset(seed=777)
rng = np.random.default_rng(31337)
rows = []
for t in range(60):
    a = [int(rng.integers(2)), int(rng.integers(3)), int(rng.integers(3))]
    obs, r, term, trunc, i2 = env.step(a)
    rows.append(dict(r=float(r), pt=float(i2['pt']), sinr=float(i2['sinr']),
                     rlf=bool(i2['rlf'])))
    if term or trunc: break
print("JSON"+json.dumps(dict(form=PE._PWR_FORM, w=float(PE.W_PWR),
      pmin=float(PE.PT_MIN), pmax=float(PE.PT_MAX), rows=rows)))
''' % (CODE, CODE)

def arm(form):
    e = dict(os.environ)
    e.pop('P2_PWR_FORM', None)
    if form: e['P2_PWR_FORM'] = form
    r = subprocess.run([sys.executable, '-c', CHILD], capture_output=True, text=True, env=e)
    if r.returncode: sys.exit(r.stdout[-1500:] + r.stderr[-1500:])
    return json.loads(r.stdout.split('JSON', 1)[1])

print('=' * 72); print('END-TO-END REWARD PROOF'); print('=' * 72)
lin = arm(None)
assert lin['form'] == 'linear', 'default is not linear'
print('  default unset -> linear  OK')
span = lin['pmax'] - lin['pmin']
for form, f in (('quad', lambda u: u*u), ('sqrt', lambda u: u**0.5)):
    a = arm(form)
    assert a['form'] == form
    assert len(a['rows']) == len(lin['rows']), 'trajectory length changed'
    worst = 0.0; nz = 0
    for rb, ra in zip(lin['rows'], a['rows']):
        assert rb['pt'] == ra['pt'] and rb['sinr'] == ra['sinr'] and rb['rlf'] == ra['rlf'], \
            'TRAJECTORY DIVERGED - the override touched more than the reward'
        u = (rb['pt'] - lin['pmin']) / span
        expect = -lin['w'] * (f(u) - u)
        worst = max(worst, abs((ra['r'] - rb['r']) - expect))
        nz += (ra['r'] != rb['r'])
    print('  %-5s trajectory identical over %d steps; reward delta matches '
          '-W*(f(u)-u) exactly, worst residual %.2e; %d/%d steps repriced'
          % (form, len(a['rows']), worst, nz, len(a['rows'])))
    assert worst < 1e-12, 'REWARD DOES NOT USE THE FORM'
    assert nz > 0, 'THE OVERRIDE IS A NO-OP IN THE REWARD'
print('=' * 72)
print('  PROVEN: the form reaches the reward, changes only the power term, and')
print('  leaves the trajectory untouched.')
