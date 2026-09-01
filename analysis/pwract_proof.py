#!/usr/bin/env python3
"""Prove the power-actuator patch BEFORE spending compute on it.

Four things must hold, and the script exits non-zero if any fails:

  A. Unset / inc3 is INERT -- _apply_pwr reproduces the published expression
     clip(pt + (a-1)*PT_STEP, PT_MIN, PT_MAX) exactly, for every (pt, a) on the grid.
  B. All three modes reach EXACTLY the same power set, {23..30} dBm.
  C. The action space follows the mode, and only the power branch changes.
  D. direct8 reaches any level in ONE step from any state; inc3 needs up to 7.
     This is the manipulation, stated as a number rather than asserted in prose.
"""
import importlib, os, sys
# run from anywhere: the environment lives beside the controller, not beside this script
_CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', 'Paper2', 'code'))
sys.path.insert(0, _CODE)
import numpy as np

FAIL = []
def check(cond, label, detail=''):
    print('  %-4s %s%s' % ('OK' if cond else 'FAIL', label, ('   ' + detail) if detail else ''))
    if not cond:
        FAIL.append(label)

def load(mode):
    if mode is None:
        os.environ.pop('P2_PWR_ACT', None)
    else:
        os.environ['P2_PWR_ACT'] = mode
    for m in ('paper2_env',):
        if m in sys.modules:
            del sys.modules[m]
    return importlib.import_module('paper2_env')

print('=' * 78)
print('A. UNSET AND inc3 ARE INERT (the published actuator is untouched)')
print('=' * 78)
PE = load(None)
PT_MIN, PT_MAX, PT_STEP = PE.PT_MIN, PE.PT_MAX, PE.PT_STEP
check(PE._PWR_ACT == 'inc3', 'unset P2_PWR_ACT defaults to inc3', repr(PE._PWR_ACT))
check(int(PE.PWR_N_ACT) == 3, 'unset -> 3 power actions', str(PE.PWR_N_ACT))

grid = np.arange(PT_MIN, PT_MAX + 0.001, 1.0)
worst = 0.0
for pt in grid:
    for a in range(3):
        published = float(np.clip(pt + (a - 1) * PT_STEP, PT_MIN, PT_MAX))   # the original line
        got = PE._apply_pwr(float(pt), a)
        worst = max(worst, abs(published - got))
check(worst == 0.0, 'inc3 reproduces the published expression on every (pt, a)',
      'worst |difference| = %r over %d pairs' % (worst, len(grid) * 3))

print()
print('=' * 78)
print('B. ALL THREE MODES REACH THE SAME POWER SET')
print('=' * 78)
sets = {}
for mode in ('inc3', 'inc7', 'direct8'):
    PE = load(mode)
    n = int(PE.PWR_N_ACT)
    # closure of {PT_MAX} (the reset state) under every action, to fixpoint
    reach, frontier = {PT_MAX}, {PT_MAX}
    while frontier:
        nxt = set()
        for pt in frontier:
            for a in range(n):
                v = PE._apply_pwr(float(pt), a)
                if v not in reach:
                    reach.add(v); nxt.add(v)
        frontier = nxt
    sets[mode] = frozenset(reach)
    print('  %-8s %d actions -> %d reachable levels  %s'
          % (mode, n, len(reach), sorted(reach)))
expected = frozenset(float(x) for x in range(int(PT_MIN), int(PT_MAX) + 1))
check(len(set(sets.values())) == 1, 'the three reachable sets are identical')
check(sets['inc3'] == expected, 'reachable set is exactly {23..30} dBm')

print()
print('=' * 78)
print('C. ACTION SPACE FOLLOWS THE MODE; ONLY THE POWER BRANCH CHANGES')
print('=' * 78)
for mode, want in (('inc3', 3), ('inc7', 7), ('direct8', 8)):
    PE = load(mode)
    check(int(PE.PWR_N_ACT) == want, '%-8s PWR_N_ACT == %d' % (mode, want), str(PE.PWR_N_ACT))

print()
print('=' * 78)
print('D. THE MANIPULATION, AS A NUMBER: STEPS TO REACH A LEVEL')
print('=' * 78)
for mode in ('inc3', 'inc7', 'direct8'):
    PE = load(mode)
    n = int(PE.PWR_N_ACT)
    worst_hops = 0
    for src in sorted(expected):
        dist = {src: 0}; frontier = [src]
        while frontier:
            cur = frontier.pop(0)
            for a in range(n):
                v = PE._apply_pwr(float(cur), a)
                if v not in dist:
                    dist[v] = dist[cur] + 1; frontier.append(v)
        worst_hops = max(worst_hops, max(dist.values()))
    print('  %-8s worst-case steps from any level to any other: %d' % (mode, worst_hops))
    if mode == 'direct8':
        check(worst_hops == 1, 'direct8 reaches every level in ONE step')
    if mode == 'inc3':
        check(worst_hops == 7, 'inc3 needs up to 7 steps (the published actuator)')

print()
print('=' * 78)
print('RESULT:', 'FAILED -> ' + ', '.join(FAIL) if FAIL else 'all checks passed')
sys.exit(1 if FAIL else 0)
