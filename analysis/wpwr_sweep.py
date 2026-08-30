#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wpwr_sweep.py -- reward-weight sensitivity sweep for the bimodal convergence structure.

THE QUESTION. paper2_env's reward is

    r = W_REL*1{link_ok} - RLF_PEN*1{rlf} - W_PWR*(pt-PT_MIN)/(PT_MAX-PT_MIN)
        - W_SPEC*(rb-RB_MIN)/(RB_MAX-RB_MIN) - W_HO*1{ho}

with W_PWR = 0.15. Holding maximum power therefore costs 0.15 per step against a link_ok
reward of 1.0, and the observed low-power basin earns the HIGHER mean return (89.38 vs 80.27).
The objective as posed prefers the nine-times-less-reliable policy. The obvious reviewer move
is: "the bimodality is an artefact of a linear power penalty against a step-shaped reliability
cliff -- retune W_PWR and it disappears."

The 120-run six-configuration sweep does not answer this. All six configurations there are
ablations of the CONTROLLER (anticipation, CVaR, PER, transplant, LSTM) and every one of them
shares identical reward weights. The reward-weight axis has never been varied.

WHAT THIS RUNS. The published controller (--ff, alpha 0.25, anticipation on, PER on), 1500
episodes, 20 seeds, at W_PWR in {0.05, 0.45} -- a factor of three either side of the published
0.15, a nine-fold range. The 0.15 arm is not re-run: models/basin_v2 (seeds 0-19) already IS
that arm, bit-for-bit, and is used as the middle point.

SEEDS. Seeds 0-19 in every arm. This is a TREATMENT MANIPULATION, not a replication, so seed
reuse is correct design (common initial conditions), unlike the failed 28 Aug confirmatory
sweep where reusing seeds under an unchanged treatment reproduced the original bit-for-bit and
carried zero information. Note the streams diverge after the first behavioural difference --
epsilon-greedy draws from the global numpy RNG once per step and episode length depends on the
policy -- so the arms share initialisation, NOT common random numbers throughout. Analyse
UNPAIRED.

WHAT IS AND IS NOT COMPARABLE ACROSS ARMS. The arms optimise different objectives, so
return_mean and cvar25 are NOT comparable between arms and must not be reported as such. The
physical quantities are: pt_dbm (defines the basin), rlf_rate, steps, rb, ho_per_ep.

GUARDS. paper2_controller.py aborts before training if the override did not take (G1), prints
the active weights into every job log (G2), and writes w_pwr into every _metrics.csv (G3).
This driver re-reads that column after every job and refuses to count a run whose recorded
weight is not the one requested.

Usage (from Paper2\\code):
    python ..\\..\\PAPER3\\proof\\wpwr_sweep.py --w_pwr 0.05,0.45 --seeds 0-19 --workers 4
    python ..\\..\\PAPER3\\proof\\wpwr_sweep.py --aggregate_only
"""
import argparse, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..', 'Paper2', 'code'))
PY = sys.executable
BASIN_THRESHOLD_DBM = 27.0        # pre-registered in the basin paper; not tuned here


def arm_dir(base, w):
    return os.path.join(base, 'wpwr_w%g' % w)


def parse_seeds(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-'); out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def read_metrics(path):
    import pandas as pd
    return pd.read_csv(path).iloc[0]


def run_one(job, a):
    w, seed = job
    d = arm_dir(a.outdir_base, w)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'model_s%d' % seed)
    mfile = out + '_metrics.csv'
    if os.path.exists(mfile) and not a.force:
        return w, seed, 'skip', 0.0, ''
    cmd = [PY, os.path.join(CODE, 'paper2_controller.py'), '--train',
           '--episodes', str(a.episodes), '--seed', str(seed), '--out', out,
           '--eval_n', str(a.eval_n), '--eval_n_train', str(a.eval_n_train),
           '--threads', '1', '--ff', '--w_pwr', repr(float(w))]
    t0 = time.time()
    with open(out + '.log', 'w') as fh:
        fh.write(' '.join(cmd) + '\n\n'); fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=CODE)
    dt = time.time() - t0
    if rc != 0 or not os.path.exists(mfile):
        return w, seed, 'FAIL rc=%s' % rc, dt, ''
    # G3 re-check: the run must have recorded the weight it was asked for
    row = read_metrics(mfile)
    if 'w_pwr' not in row.index:
        return w, seed, 'FAIL no w_pwr column (controller not patched?)', dt, ''
    if abs(float(row['w_pwr']) - float(w)) > 1e-12:
        return w, seed, 'FAIL w_pwr recorded %s, asked %s' % (row['w_pwr'], w), dt, ''
    note = 'pt %.3f dBm  RLF %.2f%%' % (float(row['pt_dbm']), float(row['rlf_rate']))
    return w, seed, 'ok', dt, note


def collect(a):
    """Every arm present on disk, including the untouched 0.15 baseline."""
    import pandas as pd, glob
    arms = {}
    # the published 0.15 arm: basin_v2 is seeds 0-19 of exactly this configuration
    base15 = os.path.join(a.outdir_base, a.baseline_dir)
    for w, d in [(0.15, base15)] + [(w, arm_dir(a.outdir_base, w)) for w in a.w_list]:
        rows = []
        for f in sorted(glob.glob(os.path.join(d, 'model_s*_metrics.csv'))):
            s = int(re.search(r'model_s(\d+)_metrics\.csv$', os.path.basename(f)).group(1))
            r = read_metrics(f)
            rec = dict(w_pwr=w, seed=s, pt_dbm=float(r['pt_dbm']), rlf_rate=float(r['rlf_rate']),
                       return_mean=float(r['return_mean']), cvar25=float(r['cvar25']),
                       steps=float(r.get('steps', float('nan'))))
            if 'w_pwr' in r.index and abs(float(r['w_pwr']) - w) > 1e-12:
                rec['MISMATCH'] = True
            rows.append(rec)
        if rows:
            arms[w] = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
    return arms


def aggregate(a):
    import numpy as np, pandas as pd
    arms = collect(a)
    if not arms:
        print('nothing to aggregate yet'); return
    print('\n' + '=' * 78)
    print('W_PWR SWEEP -- basin structure by reward weight')
    print('=' * 78)
    print('  basin label: pt_dbm < %.1f dBm  (the paper\'s threshold, not re-tuned here)' % BASIN_THRESHOLD_DBM)
    print('  NOTE return_mean/cvar25 are measured under DIFFERENT objectives across arms')
    print('       and are not comparable between rows. pt_dbm and rlf_rate are physical.\n')
    hdr = ('%-9s %4s %6s %10s %10s %9s %9s %9s' %
           ('W_PWR', 'n', 'low/n', 'low max', 'high min', 'gap dB', 'RLF low', 'RLF high'))
    print(hdr); print('-' * len(hdr))
    out_rows = []
    for w in sorted(arms):
        df = arms[w]
        if 'MISMATCH' in df.columns and df['MISMATCH'].any():
            print('  !! ARM %g HAS %d RUNS WHOSE RECORDED w_pwr DISAGREES -- excluded from the table'
                  % (w, int(df['MISMATCH'].sum())))
            df = df[df['MISMATCH'].isna()]
        low = df[df.pt_dbm < BASIN_THRESHOLD_DBM]
        high = df[df.pt_dbm >= BASIN_THRESHOLD_DBM]
        lo_max = low.pt_dbm.max() if len(low) else float('nan')
        hi_min = high.pt_dbm.min() if len(high) else float('nan')
        gap = hi_min - lo_max if (len(low) and len(high)) else float('nan')
        print('%-9g %4d %6s %10.3f %10.3f %9.3f %9.2f %9.2f' %
              (w, len(df), '%d/%d' % (len(low), len(df)), lo_max, hi_min, gap,
               low.rlf_rate.mean() if len(low) else float('nan'),
               high.rlf_rate.mean() if len(high) else float('nan')))
        out_rows.append(dict(w_pwr=w, n=len(df), n_low=len(low), low_max=lo_max, high_min=hi_min,
                             empty_gap_db=gap,
                             rlf_low=low.rlf_rate.mean() if len(low) else float('nan'),
                             rlf_high=high.rlf_rate.mean() if len(high) else float('nan'),
                             pt_min=df.pt_dbm.min(), pt_max=df.pt_dbm.max(),
                             n_between=int(((df.pt_dbm > 25) & (df.pt_dbm < 28)).sum())))
    print('\n  runs landing strictly between 25 and 28 dBm (the paper\'s empty interval):')
    for r in out_rows:
        print('    W_PWR=%-6g %d of %d' % (r['w_pwr'], r['n_between'], r['n']))
    allp = pd.concat(list(arms.values()), ignore_index=True)
    os.makedirs(os.path.join(CODE, 'results'), exist_ok=True)
    p1 = os.path.join(CODE, 'results', 'wpwr_sweep_runs.csv')
    p2 = os.path.join(CODE, 'results', 'wpwr_sweep_summary.csv')
    allp.to_csv(p1, index=False); pd.DataFrame(out_rows).to_csv(p2, index=False)
    print('\nsaved -> %s\n         %s' % (p1, p2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w_pwr', default='0.05,0.45', help='weights to RUN (0.15 is taken from basin_v2)')
    ap.add_argument('--seeds', default='0-19')
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--outdir_base', default=os.path.join(CODE, 'models'))
    ap.add_argument('--baseline_dir', default='basin_v2', help='the existing W_PWR=0.15 seeds 0-19 arm')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate_only', action='store_true')
    a = ap.parse_args()
    a.w_list = [float(x) for x in a.w_pwr.split(',') if x.strip()]

    if a.aggregate_only:
        aggregate(a); return

    # ---- pre-flight, before a single episode ----------------------------------------
    ctl = os.path.join(CODE, 'paper2_controller.py')
    env = os.path.join(CODE, 'paper2_env.py')
    src_c, src_e = open(ctl, encoding='utf-8').read(), open(env, encoding='utf-8').read()
    if '--w_pwr' not in src_c or 'w_pwr=float(PE.W_PWR)' not in src_c:
        sys.exit('PRE-FLIGHT FAIL: paper2_controller.py is not patched. Run patch_wpwr_20260830.py.')
    if '_W_PWR_ENV' not in src_e:
        sys.exit('PRE-FLIGHT FAIL: paper2_env.py is not patched. Run patch_wpwr_20260830.py.')
    base = os.path.join(a.outdir_base, a.baseline_dir)
    if not os.path.isdir(base):
        print('PRE-FLIGHT WARN: baseline arm %s missing; the 0.15 row will be absent.' % base)
    cache = os.path.join(CODE, 'geom_cache')
    n_cached = len([f for f in os.listdir(cache)]) if os.path.isdir(cache) else 0
    if n_cached < 45:
        sys.exit('PRE-FLIGHT FAIL: geom_cache holds %d/45 scenes. Parallel workers would race on '
                 'the pickle writes. Warm it single-threaded first.' % n_cached)
    seeds = parse_seeds(a.seeds)
    jobs = [(w, s) for w in a.w_list for s in seeds]
    todo = [j for j in jobs
            if a.force or not os.path.exists(os.path.join(arm_dir(a.outdir_base, j[0]),
                                                          'model_s%d_metrics.csv' % j[1]))]
    print('PRE-FLIGHT OK: both files patched, geom_cache %d/45 warm (no write race)' % n_cached)
    print('W_PWR sweep | arms %s | seeds %s | %d jobs, %d to run | %d workers | %d episodes'
          % (a.w_list, a.seeds, len(jobs), len(todo), a.workers, a.episodes))
    print('  baseline W_PWR=0.15 taken from %s (not re-run)' % base)
    print('  est. ~5 min/run at 1 worker; expect roughly %.1f h wall at %d workers\n'
          % (len(todo) * 5.0 / 60.0 / max(a.workers * 0.8, 1), a.workers), flush=True)

    t0, done, fails = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for w, seed, st, dt, note in ex.map(lambda j: run_one(j, a), jobs):
            if st == 'skip':
                continue
            done += 1
            if st != 'ok':
                fails.append((w, seed, st))
            el = time.time() - t0
            eta = el / done * (len(todo) - done)
            print('[%d/%d] w=%g s%d: %s in %.1f min  %s | elapsed %.0f min, eta %.0f min'
                  % (done, len(todo), w, seed, st, dt / 60, note, el / 60, eta / 60), flush=True)

    print('\nsweep finished in %.1f min' % ((time.time() - t0) / 60))
    if fails:
        print('!! %d FAILED JOBS -- results below are INCOMPLETE:' % len(fails))
        for w, s, st in fails:
            print('   w=%g s%d: %s' % (w, s, st))
    aggregate(a)


if __name__ == '__main__':
    main()
