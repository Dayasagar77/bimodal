#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_pwrform_20260830.py -- add a POWER-PENALTY FORM override to the Paper-2 env.

WHY. Vamplew et al. (2008) show linear scalarisation cannot recover Pareto-optimal
policies in non-convex regions of the front: two separated policies can tie in
scalarised value with nothing achievable between them. The reward here is exactly a
linear scalarisation of availability against power, so that is the best available
account of an empty interval. Sec IV-E varied the WEIGHT of that penalty over a
22-fold range without dissolving the structure; its FORM has never been varied.

THE DESIGN PROPERTY THAT MAKES THIS A CLEAN TEST. With u = (Pt - Pmin)/(Pmax - Pmin),
every form below satisfies f(0) = 0 and f(1) = 1. So the two RAIL policies -- the ones
the paper reports -- receive IDENTICAL reward under every form, and only INTERIOR
(intermediate-power) policies are repriced. A change in outcome therefore cannot be
attributed to the rails being made cheaper or dearer; it can only be attributed to the
interior becoming reachable or not.

    linear  f(u) = u        the published reward
    quad    f(u) = u^2      interior CHEAPER than linear  (convex penalty)
    sqrt    f(u) = sqrt(u)  interior DEARER  than linear  (concave penalty)

Guards, as for the weight override: the trainer aborts before training if the form did
not take; the active form prints into every job log; and it is written into every
_metrics.csv so a silent no-op is visible in the data rather than merely absent from it.

Default unchanged: P2_PWR_FORM unset -> linear -> every stored result stays valid.
Idempotent.
"""
import os, re, shutil, sys

CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'Paper2', 'code'))
STAMP = '20260830'

def patch(path, anchor, insert, marker):
    src = open(path, encoding='utf-8').read()
    if marker in src:
        print('  SKIP  %s: already patched' % os.path.basename(path)); return
    n = src.count(anchor)
    if n != 1:
        sys.exit('  ABORT %s: anchor matched %d times, expected 1' % (os.path.basename(path), n))
    bak = path + '.bak_pwrform_%s' % STAMP
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    open(path, 'w', encoding='utf-8').write(src.replace(anchor, insert, 1))
    print('  OK    %s patched (backup %s)' % (os.path.basename(path), os.path.basename(bak)))

env_p = os.path.join(CODE, 'paper2_env.py')
A1 = "_W_PWR_ENV = os.environ.get('P2_W_PWR')"
I1 = """# --- power-penalty FORM override, added 30 Aug 2026. Every form is normalised so
# f(0)=0 and f(1)=1: the two rail policies are priced identically under all of them,
# and only intermediate-power policies are repriced. See patch_pwrform_20260830.py.
_PWR_FORM = os.environ.get('P2_PWR_FORM', 'linear')
if _PWR_FORM not in ('linear', 'quad', 'sqrt'):
    raise ValueError('P2_PWR_FORM must be linear, quad or sqrt; got %r' % _PWR_FORM)

def _pwr_cost(u):
    \"\"\"Normalised transmit-power cost. u in [0,1]; f(0)=0 and f(1)=1 for every form.\"\"\"
    if _PWR_FORM == 'quad':
        return u * u
    if _PWR_FORM == 'sqrt':
        return u ** 0.5
    return u
# ----------------------------------------------------------------------------------
_W_PWR_ENV = os.environ.get('P2_W_PWR')"""
patch(env_p, A1, I1, '_PWR_FORM')

A2 = "- W_PWR * (self.pt - PT_MIN) / (PT_MAX - PT_MIN)"
I2 = "- W_PWR * _pwr_cost((self.pt - PT_MIN) / (PT_MAX - PT_MIN))"
patch(env_p, A2, I2, '_pwr_cost((self.pt')

ctl_p = os.path.join(CODE, 'paper2_controller.py')
A3 = ("    ap.add_argument('--w_pwr', type=float, default=None,")
I3 = ("    ap.add_argument('--pwr_form', default=None, choices=['linear', 'quad', 'sqrt'],\n"
      "                    help='shape of the transmit-power penalty in the reward. All forms are\\n'\n"
      "                         'normalised so the rail policies price identically; only interior\\n'\n"
      "                         'power levels are repriced. Mechanism test, 30 Aug 2026.')\n"
      "    ap.add_argument('--w_pwr', type=float, default=None,")
patch(ctl_p, A3, I3, '--pwr_form')

A4 = "    if a.w_pwr is not None:\n        os.environ['P2_W_PWR'] = repr(float(a.w_pwr))"
I4 = ("    if a.pwr_form is not None:\n        os.environ['P2_PWR_FORM'] = a.pwr_form\n"
      "    if a.w_pwr is not None:\n        os.environ['P2_W_PWR'] = repr(float(a.w_pwr))")
patch(ctl_p, A4, I4, "os.environ['P2_PWR_FORM']")

A5 = "    if a.w_pwr is not None and abs(PE.W_PWR - a.w_pwr) > 1e-12:"
I5 = ("    if a.pwr_form is not None and PE._PWR_FORM != a.pwr_form:\n"
      "        sys.exit(f'ABORT: asked for pwr_form={a.pwr_form!r} but paper2_env._PWR_FORM is '\n"
      "                 f'{PE._PWR_FORM!r}. The override did not take -- refusing to train.')\n"
      "    if a.w_pwr is not None and abs(PE.W_PWR - a.w_pwr) > 1e-12:")
patch(ctl_p, A5, I5, 'PE._PWR_FORM != a.pwr_form')

A6 = "per=int(not a.no_per), episodes=a.episodes, w_pwr=float(PE.W_PWR),"
I6 = "per=int(not a.no_per), episodes=a.episodes, w_pwr=float(PE.W_PWR), pwr_form=PE._PWR_FORM,"
patch(ctl_p, A6, I6, 'pwr_form=PE._PWR_FORM')

A7 = 'f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO}"'
I7 = 'f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO} form={PE._PWR_FORM}"'
patch(ctl_p, A7, I7, 'form={PE._PWR_FORM}')
print('\npatch complete.')
