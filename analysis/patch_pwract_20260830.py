#!/usr/bin/env python3
"""Add a SELECTABLE TRANSMIT-POWER ACTUATOR to paper2_env.py / paper2_controller.py.

Idempotent, backs up both files. Adds P2_PWR_ACT with three modes:

    inc3     the published actuator, unchanged: 3 actions, delta {-1,0,+1} dB, then clip
    inc7     same family, wider steps:          7 actions, delta {-3..+3} dB, then clip
    direct8  absolute levels:                   8 actions, a -> 23+a dBm in ONE step

All three reach EXACTLY the same power set, {23,...,30} dBm on a 1 dB grid, because the
published actuator starts every episode at PT_MAX and moves in 1 dB steps under clipping.
A difference in outcome therefore cannot be attributed to a different set of achievable
powers -- only to how many steps it takes to reach one.

Read at MODULE SCOPE, like P2_W_PWR and P2_PWR_FORM, because paper2ma_env.py copies these
names at import time and a post-import assignment would silently fail to propagate.
"""
import re, shutil, sys, os

D = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(D, '..', '..', 'Paper2', 'code'))
ENV = os.path.join(CODE, 'paper2_env.py')
CTL = os.path.join(CODE, 'paper2_controller.py')
TAG = '_PWR_ACT'

ENV_BLOCK = '''
# --- POWER ACTUATOR parameterisation, added 30 Aug 2026. -------------------------------------
# The remaining candidate explanation of the empty 25-28 dBm interval is the actuator itself:
# the published power branch is a three-action increment/decrement control with hard limits,
# and a deterministic greedy policy over such a branch tends to saturate at a limit however the
# interior is priced. Testing that needs the ADDRESSING changed while the reachable power set is
# held fixed. Every mode below lands on the same 1 dB grid over [PT_MIN, PT_MAX].
_PWR_ACT = os.environ.get('P2_PWR_ACT', 'inc3')
if _PWR_ACT not in ('inc3', 'inc7', 'direct8'):
    raise ValueError('P2_PWR_ACT must be inc3, inc7 or direct8; got %r' % _PWR_ACT)

_PWR_DELTAS = {'inc3': (-1.0, 0.0, 1.0),
               'inc7': (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)}
PWR_N_ACT = 8 if _PWR_ACT == 'direct8' else len(_PWR_DELTAS[_PWR_ACT])


def _apply_pwr(pt, a):
    """Next transmit power (dBm) from the current power and the power-branch action.

    inc3    - published: a in {0,1,2} -> delta {-1,0,+1} dB, then clip.  a-1 exactly
              reproduces the original expression, so an unset P2_PWR_ACT is inert.
    inc7    - same family, wider steps: a in {0..6} -> delta {-3..+3} dB, then clip.
              Included as the CONTROL for branch width and traversal speed: it shares
              direct8's faster traversal and near-equal action count while remaining
              incremental, so a difference between inc7 and direct8 isolates addressing.
    direct8 - absolute: a in {0..7} -> PT_MIN + a dBm, reached in a single step.
    """
    if _PWR_ACT == 'direct8':
        return float(PT_MIN + a)
    return float(np.clip(pt + _PWR_DELTAS[_PWR_ACT][a] * PT_STEP, PT_MIN, PT_MAX))
'''

CTL_ARG = """    ap.add_argument('--pwr_act', default=None, choices=['inc3', 'inc7', 'direct8'],
                    help='transmit-power actuator parameterisation. All modes share the\\n'
                         'same reachable power set {23..30} dBm; they differ only in how\\n'
                         'many steps it takes to reach a level. Mechanism test, 30 Aug 2026.')
"""


def patch(path, edits, tag):
    src = open(path, encoding='utf-8').read()
    if tag in src:
        print('  already patched: %s' % os.path.basename(path)); return False
    bak = path + '.bak_pwract_20260830'
    if not os.path.exists(bak):
        shutil.copy2(path, bak); print('  backup -> %s' % os.path.basename(bak))
    for old, new, label in edits:
        n = src.count(old)
        if n != 1:
            sys.exit('ABORT: anchor %r matched %d times in %s -- refusing to patch'
                     % (label, n, os.path.basename(path)))
        src = src.replace(old, new)
        print('    ok  %s' % label)
    open(path, 'w', encoding='utf-8').write(src)
    print('  patched: %s' % os.path.basename(path))
    return True


print('ENV  %s' % ENV)
patch(ENV, [
    ("_W_PWR_ENV = os.environ.get('P2_W_PWR')\nif _W_PWR_ENV is not None:\n    W_PWR = float(_W_PWR_ENV)",
     "_W_PWR_ENV = os.environ.get('P2_W_PWR')\nif _W_PWR_ENV is not None:\n    W_PWR = float(_W_PWR_ENV)\n" + ENV_BLOCK,
     'actuator block inserted at module scope'),
    ("self.action_space = spaces.MultiDiscrete([2, 3, 3])",
     "self.action_space = spaces.MultiDiscrete([2, PWR_N_ACT, 3])",
     'action space follows the mode'),
    ("self.pt = float(np.clip(self.pt + (a_pwr - 1) * PT_STEP, PT_MIN, PT_MAX))",
     "self.pt = _apply_pwr(self.pt, a_pwr)",
     'power update routed through _apply_pwr'),
    ("; action: branched [HO(2),pwr(3),spec(3)]",
     "; action: branched [HO(2),pwr(PWR_N_ACT),spec(3)]",
     'header comment'),
], TAG)

print('CTL  %s' % CTL)
patch(CTL, [
    ("    ap.add_argument('--pwr_form', default=None, choices=['linear', 'quad', 'sqrt'],",
     CTL_ARG + "    ap.add_argument('--pwr_form', default=None, choices=['linear', 'quad', 'sqrt'],",
     'argparse --pwr_act'),
    ("    if a.pwr_form is not None:\n        os.environ['P2_PWR_FORM'] = a.pwr_form",
     "    if a.pwr_act is not None:\n        os.environ['P2_PWR_ACT'] = a.pwr_act\n"
     "    if a.pwr_form is not None:\n        os.environ['P2_PWR_FORM'] = a.pwr_form",
     'env var set BEFORE the env import'),
    ("    if a.pwr_form is not None and PE._PWR_FORM != a.pwr_form:",
     "    # G1a: the actuator must have taken, AND the controller's branch widths must follow it.\n"
     "    if a.pwr_act is not None and PE._PWR_ACT != a.pwr_act:\n"
     "        sys.exit(f'ABORT: asked for pwr_act={a.pwr_act!r} but paper2_env._PWR_ACT is '\n"
     "                 f'{PE._PWR_ACT!r}. The override did not take -- refusing to train.')\n"
     "    BRANCHES[1] = int(PE.PWR_N_ACT)          # in-place: paper2ma_train binds the same list\n"
     "    _nvec = list(PE.MultiRSUBlockageEnv.__init__.__doc__ and [] or [])  # no-op, keeps linters quiet\n"
     "    if a.pwr_form is not None and PE._PWR_FORM != a.pwr_form:",
     'abort guard + BRANCHES follows the env'),
    ("w_pwr=float(PE.W_PWR), pwr_form=PE._PWR_FORM,",
     "w_pwr=float(PE.W_PWR), pwr_form=PE._PWR_FORM, pwr_act=PE._PWR_ACT,\n"
     "             pwr_n_act=int(PE.PWR_N_ACT),",
     'metrics record the actuator'),
    ('f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO} form={PE._PWR_FORM}"',
     'f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO} form={PE._PWR_FORM} act={PE._PWR_ACT}({PE.PWR_N_ACT})"',
     'config banner'),
], TAG)
print('done.')
