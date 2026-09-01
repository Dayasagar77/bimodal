#!/usr/bin/env python3
"""Power-ACTUATOR sweep: does the empty 25-28 dBm interval survive direct addressing?

Two new arms, twenty seeds each. The published actuator (inc3) is runs/basin_v2, already on
disk and NOT re-run.

    inc3     3 actions, delta {-1,0,+1} dB   worst-case 7 steps to cross the range  [baseline]
    inc7     7 actions, delta {-3..+3} dB    worst-case 3 steps                     [control]
    direct8  8 actions, absolute 23..30 dBm  worst-case 1 step                      [treatment]

All three reach exactly the same power set. inc7 is the control for branch width and traversal
speed: it shares direct8's fast traversal and near-equal action count while remaining
incremental, so a direct8-vs-inc7 difference isolates ADDRESSING from width and speed.

Usage (from Paper2\\code):
    python ..\\..\\PAPER3\\proof\\pwract_sweep.py --acts direct8,inc7 --seeds 0-19 --workers 4
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', 'Paper2', 'code'))
PY = sys.executable
THRESH = 27.0


def parse_seeds(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-'); out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def run_one(job, a):
    act, seed = job
    d = os.path.join(a.outdir_base, 'pwract_' + act)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'model_s%d' % seed)
    mfile = out + '_metrics.csv'
    if os.path.exists(mfile) and not a.force:
        return act, seed, 'skip', 0.0, ''
    cmd = [PY, os.path.join(CODE, 'paper2_controller.py'), '--train',
           '--episodes', str(a.episodes), '--seed', str(seed), '--out', out,
           '--eval_n', str(a.eval_n), '--eval_n_train', str(a.eval_n_train),
           '--threads', '1', '--ff', '--pwr_act', act]
    t0 = time.time()
    with open(out + '.log', 'w') as fh:
        fh.write(' '.join(cmd) + '\n\n'); fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=CODE)
    dt = time.time() - t0
    if rc != 0 or not os.path.exists(mfile):
        return act, seed, 'FAIL rc=%s' % rc, dt, ''
    import pandas as pd
    r = pd.read_csv(mfile).iloc[0]
    # G2: every result file must state the actuator it was trained under, and it must match.
    if 'pwr_act' not in r.index:
        return act, seed, 'FAIL no pwr_act column', dt, ''
    if str(r['pwr_act']) != act:
        return act, seed, 'FAIL act mismatch %s' % r['pwr_act'], dt, ''
    # rlf_rate is stored as a percentage already -- do not scale it again
    return (act, seed, 'ok', dt,
            'pt %6.3f dBm  RLF %.2f%%' % (float(r['pt_dbm']), float(r['rlf_rate'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', default='direct8,inc7')
    ap.add_argument('--seeds', default='0-19')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--outdir_base', default=os.path.join(CODE, 'models'))
    a = ap.parse_args()
    a.act_list = [x.strip() for x in a.acts.split(',') if x.strip()]
    for act in a.act_list:
        if act not in ('inc3', 'inc7', 'direct8'):
            sys.exit('unknown actuator %r' % act)
    seeds = parse_seeds(a.seeds)

    # --- pre-flight: refuse to start unless the patch is in and proven -------------
    sys.path.insert(0, CODE)
    import importlib
    for mode in a.act_list:
        os.environ['P2_PWR_ACT'] = mode
        for m in ('paper2_env',):
            sys.modules.pop(m, None)
        PE = importlib.import_module('paper2_env')
        if PE._PWR_ACT != mode:
            sys.exit('PRE-FLIGHT FAIL: P2_PWR_ACT=%s did not take (got %s)' % (mode, PE._PWR_ACT))
    os.environ.pop('P2_PWR_ACT', None)
    print('PRE-FLIGHT OK: actuator override responds for %s' % ', '.join(a.act_list))

    jobs = [(act, s) for act in a.act_list for s in seeds]
    todo = [j for j in jobs if a.force or not os.path.exists(
        os.path.join(a.outdir_base, 'pwract_' + j[0], 'model_s%d_metrics.csv' % j[1]))]
    print('actuator sweep | %s | seeds %d-%d | %d jobs, %d to run | %d workers'
          % (a.act_list, seeds[0], seeds[-1], len(jobs), len(todo), a.workers), flush=True)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for act, seed, status, dt, extra in ex.map(lambda j: run_one(j, a), todo):
            done += 1
            el = (time.time() - t0) / 60.0
            eta = el / done * (len(todo) - done)
            print('[%d/%d] %s s%d: %s in %.1f min  %s | elapsed %.0f min, eta %.0f min'
                  % (done, len(todo), act, seed, status, dt / 60.0, extra, el, eta), flush=True)

    # --- summary -------------------------------------------------------------------
    import pandas as pd
    print('\n' + '=' * 78)
    print('POWER-ACTUATOR SWEEP -- does direct addressing populate 25-28 dBm?')
    print('=' * 78)
    print('%-9s %4s %6s %9s %9s %9s %9s' %
          ('actuator', 'n', 'low/n', 'IN 25-28', 'low max', 'high min', 'gap dB'))
    print('-' * 68)
    rows = [('inc3', os.path.join(a.outdir_base, 'basin_v2'))] + \
           [(x, os.path.join(a.outdir_base, 'pwract_' + x)) for x in a.act_list]
    for act, d in rows:
        pts, mism = [], 0
        for s in seeds:
            f = os.path.join(d, 'model_s%d_metrics.csv' % s)
            if not os.path.exists(f):
                continue
            r = pd.read_csv(f).iloc[0]
            if 'pwr_act' in r.index and act != 'inc3' and str(r['pwr_act']) != act:
                mism += 1
            pts.append(float(r['pt_dbm']))
        if not pts:
            continue
        pts = pd.Series(pts)
        lo, hi = pts[pts < THRESH], pts[pts >= THRESH]
        print('%-9s %4d %6s %9d %9s %9s %9s'
              % (act, len(pts), '%d/%d' % (len(lo), len(pts)),
                 int(((pts > 25) & (pts < 28)).sum()),
                 ('%.3f' % lo.max()) if len(lo) else '-',
                 ('%.3f' % hi.min()) if len(hi) else '-',
                 ('%.3f' % (hi.min() - lo.max())) if len(lo) and len(hi) else '-'))
        if mism:
            print('   !! %d actuator mismatches in %s' % (mism, act))
    print('\nQ1 is decided by the "IN 25-28" column for direct8; inc7 separates addressing')
    print('from branch width and traversal speed.')


if __name__ == '__main__':
    main()
