#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pwrform_sweep.py -- does the FORM of the power penalty create the empty interval?

Pre-registered 30 Aug 2026 (DnDQN_Paper3_PwrForm_PREREGISTRATION_20260830). Every form
is normalised so f(0)=0 and f(1)=1, so the two rail policies price identically and only
interior power levels are repriced: quad cheaper, sqrt dearer. A change in outcome is
therefore attributable to the interior alone.

The `linear` arm is models/basin_v2 -- already on disk, not re-run.

GUARDS, as for the weight sweep: the trainer aborts before training if the form did not
take; the active form prints into every job log; and pwr_form is written into every
_metrics.csv, which this driver re-reads after each job and refuses to count on mismatch.

Usage (from Paper2\\code):
    python ..\\..\\PAPER3\\proof\\pwrform_sweep.py --forms quad,sqrt --seeds 0-19 --workers 4
"""
import argparse, os, re, subprocess, sys, time
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
    form, seed = job
    d = os.path.join(a.outdir_base, 'pwrform_' + form)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'model_s%d' % seed)
    mfile = out + '_metrics.csv'
    if os.path.exists(mfile) and not a.force:
        return form, seed, 'skip', 0.0, ''
    cmd = [PY, os.path.join(CODE, 'paper2_controller.py'), '--train',
           '--episodes', str(a.episodes), '--seed', str(seed), '--out', out,
           '--eval_n', str(a.eval_n), '--eval_n_train', str(a.eval_n_train),
           '--threads', '1', '--ff', '--pwr_form', form]
    t0 = time.time()
    with open(out + '.log', 'w') as fh:
        fh.write(' '.join(cmd) + '\n\n'); fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=CODE)
    dt = time.time() - t0
    if rc != 0 or not os.path.exists(mfile):
        return form, seed, 'FAIL rc=%s' % rc, dt, ''
    import pandas as pd
    r = pd.read_csv(mfile).iloc[0]
    if 'pwr_form' not in r.index:
        return form, seed, 'FAIL no pwr_form column (controller not patched?)', dt, ''
    if str(r['pwr_form']) != form:
        return form, seed, 'FAIL recorded form %r, asked %r' % (r['pwr_form'], form), dt, ''
    pt = float(r['pt_dbm'])
    mid = ' *** IN THE INTERVAL ***' if 25.0 < pt < 28.0 else ''
    return form, seed, 'ok', dt, 'pt %.3f dBm  RLF %.2f%%%s' % (pt, float(r['rlf_rate']), mid)


def aggregate(a):
    import pandas as pd, glob
    print('\n' + '=' * 74)
    print('POWER-PENALTY FORM SWEEP -- Q1: does any run land in 25-28 dBm?')
    print('=' * 74)
    hdr = '%-9s %4s %7s %9s %10s %10s %9s' % ('form', 'n', 'low/n', 'IN 25-28', 'low max',
                                              'high min', 'gap dB')
    print(hdr); print('-' * len(hdr))
    for form, d in [('linear', os.path.join(a.outdir_base, 'basin_v2'))] + \
                   [(f, os.path.join(a.outdir_base, 'pwrform_' + f)) for f in a.form_list]:
        rows = []
        for f in sorted(glob.glob(os.path.join(d, 'model_s*_metrics.csv'))):
            r = pd.read_csv(f).iloc[0]
            rows.append((float(r['pt_dbm']), float(r['rlf_rate'])))
        if not rows:
            continue
        pt = [x[0] for x in rows]
        lo = [x for x in rows if x[0] < THRESH]; hi = [x for x in rows if x[0] >= THRESH]
        mid = sum(1 for p in pt if 25.0 < p < 28.0)
        gap = (min(x[0] for x in hi) - max(x[0] for x in lo)) if lo and hi else float('nan')
        print('%-9s %4d %7s %9d %10.3f %10.3f %9.3f'
              % (form, len(rows), '%d/%d' % (len(lo), len(rows)), mid,
                 max(x[0] for x in lo) if lo else float('nan'),
                 min(x[0] for x in hi) if hi else float('nan'), gap))
    print('\nQ1 is decided by the "IN 25-28" column: any non-zero entry for quad')
    print('implicates the linear form; all zeros rules linear scalarisation out too.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forms', default='quad,sqrt')
    ap.add_argument('--seeds', default='0-19')
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--outdir_base', default=os.path.join(CODE, 'models'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate_only', action='store_true')
    a = ap.parse_args()
    a.form_list = [f.strip() for f in a.forms.split(',') if f.strip()]

    if a.aggregate_only:
        aggregate(a); return

    src = open(os.path.join(CODE, 'paper2_env.py'), encoding='utf-8').read()
    if '_PWR_FORM' not in src or '_pwr_cost((self.pt' not in src:
        sys.exit('PRE-FLIGHT FAIL: paper2_env.py not patched. Run patch_pwrform_20260830.py.')
    ctl = open(os.path.join(CODE, 'paper2_controller.py'), encoding='utf-8').read()
    if '--pwr_form' not in ctl or 'pwr_form=PE._PWR_FORM' not in ctl:
        sys.exit('PRE-FLIGHT FAIL: paper2_controller.py not patched.')
    cache = os.path.join(CODE, 'geom_cache')
    n = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    if n < 45:
        sys.exit('PRE-FLIGHT FAIL: geom_cache %d/45 -- parallel workers would race.' % n)

    seeds = parse_seeds(a.seeds)
    jobs = [(f, s) for f in a.form_list for s in seeds]
    todo = [j for j in jobs if a.force or not os.path.exists(
        os.path.join(a.outdir_base, 'pwrform_' + j[0], 'model_s%d_metrics.csv' % j[1]))]
    print('PRE-FLIGHT OK: both files patched, geom_cache %d/45 warm' % n)
    print('form sweep | %s | seeds %s | %d jobs, %d to run | %d workers'
          % (a.form_list, a.seeds, len(jobs), len(todo), a.workers), flush=True)

    t0, done, fails = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for form, seed, st, dt, note in ex.map(lambda j: run_one(j, a), jobs):
            if st == 'skip':
                continue
            done += 1
            if st != 'ok':
                fails.append((form, seed, st))
            el = time.time() - t0
            print('[%d/%d] %s s%d: %s in %.1f min  %s | elapsed %.0f min, eta %.0f min'
                  % (done, len(todo), form, seed, st, dt / 60, note, el / 60,
                     el / done * (len(todo) - done) / 60), flush=True)
    print('\nfinished in %.1f min' % ((time.time() - t0) / 60))
    for f in fails:
        print('  FAILED:', f)
    aggregate(a)


if __name__ == '__main__':
    main()
