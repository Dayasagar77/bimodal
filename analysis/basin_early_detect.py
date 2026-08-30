#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""basin_early_detect.py — is the convergence basin predictable EARLY in training?

WHY THIS QUESTION. Ranked by measured effect size, the convergence basin is the largest
CONTROLLABLE lever in this programme: 8.90% sustained RLF in the low-power basin against
0.96% in the high-power one, a 9x difference. Admission capacity is bigger (235x) but sits
outside the action space. CVaR selection is worth -0.2 pp. The spectrum actuator is worth
zero. We spent four candidates on the small effects and never touched the large one.

    Does the EARLY return trajectory predict which basin a run ends in?

  yes -> a detector exists, and "steer training into the reliable basin" becomes a real
         mechanism with a 9x target and 130 labelled runs to fit against.
  no  -> per-episode return does not carry it. The next cheapest step is a one-line change
         to log mean transmit power per episode, then retrain on FIST. Bounded, not a guess.

NO NEW TRAINING. Reads models/sweep_r3/*_returns.csv (1500 episodes each) and
all_metrics.csv (final pt_dbm = the basin label). 130 runs, 41 low / 89 high.

PRE-REGISTERED DECISION RULE (written before the first run, 28 Aug 2026):

    WITHIN-CONFIG mean AUC >= 0.80, at a window ending at or before episode 500 (one third
    of training), with family-wise permutation p < 0.05
        -> a detector is plausible. Run the prior-art gate BEFORE designing anything.
    otherwise
        -> return does not carry the basin. Do not build a detector on it.

THE CONFOUND THIS SCRIPT IS BUILT AROUND. Configurations differ in basin propensity --
6/20 low for the reported controller, 13/20 for -CVaR -- and they also differ in return.
Pooling all 130 runs would let that difference masquerade as early detectability, which is
the same confound class that produced a spurious 5.5x effect in the Candidate D pre-check
and a spurious seed correlation in Candidate B. So the verdict uses the WITHIN-CONFIG AUC,
with labels permuted inside each config. The naive pooled figure is printed alongside,
labelled, so the size of the confound is visible rather than hidden.

MULTIPLICITY. Several windows x several features are tried. The permutation test uses the
MAX statistic across all of them, which controls family-wise error exactly -- rather than
reporting the best of N uncorrected p-values, which is how a null becomes a finding.

Usage (from Paper2\\code):
    python "..\\..\\PAPER3\\proof\\basin_early_detect.py"
    python "..\\..\\PAPER3\\proof\\basin_early_detect.py" --dir models/sweep_r3 --iters 20000
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np


def auc(x, y):
    """Rank-based AUC: P(x_pos > x_neg). y is the boolean label. No sklearn."""
    x = np.asarray(x, float); y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(x)) + 1.0
    return (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


FEATURES = ['mean', 'std', 'slope', 'last_quarter_mean', 'max', 'min']


def feats(ret, lo, hi):
    v = ret[lo:hi]
    if len(v) < 10:
        return None
    x = np.arange(len(v), dtype=float)
    q = v[int(0.75 * len(v)):]
    return dict(mean=float(v.mean()), std=float(v.std()),
                slope=float(np.polyfit(x, v, 1)[0]) * 100.0,
                last_quarter_mean=float(q.mean()),
                max=float(v.max()), min=float(v.min()))


def load(d, thr):
    """Join per-episode series to the final basin label.

    Reads all_metrics.csv when present; otherwise falls back to the per-seed
    <config>_s<seed>_metrics.csv files that a fresh sweep writes, deriving the config
    from the filename. Returns every numeric per-episode column found, so a run that
    logged pt_mean is analysed on power as well as on return -- power being the quantity
    that DEFINES the basin, which return only proxies."""
    meta = {}
    am = os.path.join(d, 'all_metrics.csv')
    if os.path.exists(am):
        for r in csv.DictReader(open(am)):
            meta[(r['config'], int(r['seed']))] = float(r['pt_dbm'])
    else:
        for p in sorted(glob.glob(os.path.join(d, '*_metrics.csv'))):
            m = re.match(r'(.+)_s(\d+)_metrics\.csv$', os.path.basename(p))
            if not m:
                continue
            for r in csv.DictReader(open(p)):
                meta[(m.group(1), int(m.group(2)))] = float(r['pt_dbm'])
                break
        if meta:
            print(f"  (no all_metrics.csv; read {len(meta)} per-seed *_metrics.csv files)")

    runs = []
    for p in sorted(glob.glob(os.path.join(d, '*_returns.csv'))):
        m = re.match(r'(.+)_s(\d+)_returns\.csv$', os.path.basename(p))
        if not m:
            continue
        cfg, sd = m.group(1), int(m.group(2))
        if (cfg, sd) not in meta:
            continue
        rd = list(csv.DictReader(open(p)))
        if not rd:
            continue
        series = {}
        for col in rd[0]:
            if col == 'episode':
                continue
            try:
                v = np.array([float(r[col]) for r in rd])
            except ValueError:
                continue                       # non-numeric column (e.g. term=True/False)
            if np.isfinite(v).sum() >= 10:
                series[col] = v
        if 'return' not in series:
            continue
        runs.append(dict(config=cfg, seed=sd, series=series,
                         pt=meta[(cfg, sd)], low=meta[(cfg, sd)] < thr))
    return runs


def precompute(runs, windows, observables):
    """Feature matrix, computed ONCE, for every (observable, window, feature).

    The permutation test only reshuffles labels, so recomputing polyfit inside the loop is
    pure waste -- 30 cells x 130 runs x 20,000 iterations is 78 million polyfits and turns
    a seconds-long job into an hours-long one."""
    X = {}
    for obs in observables:
        for lo, hi in windows:
            cols = {f: np.full(len(runs), np.nan) for f in FEATURES}
            for i, r in enumerate(runs):
                v = feats(r['series'][obs], lo, hi) if obs in r['series'] else None
                if v is None:
                    continue
                for f in FEATURES:
                    cols[f][i] = v[f]
            for f in FEATURES:
                X[(obs, lo, hi, f)] = cols[f]
    return X


def wc_auc(x, labels, groups):
    """Within-config AUC from a precomputed column, averaged weighted by usable n."""
    num, den = 0.0, 0
    for idx in groups:
        xi, yi = x[idx], labels[idx]
        ok = np.isfinite(xi)
        xi, yi = xi[ok], yi[ok]
        if yi.sum() < 3 or (~yi).sum() < 3:
            continue                                  # too few of one class in this config
        a = auc(xi, yi)
        if np.isfinite(a):
            num += a * len(yi); den += len(yi)
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='models/sweep_r3')
    ap.add_argument('--threshold', type=float, default=27.0,
                    help='dBm cut. The basin analysis showed any cut in 24-29 classifies all '
                         '120 runs identically, so this is not a tunable knob.')
    ap.add_argument('--windows', default='100,200,300,500,750')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--auc_target', type=float, default=0.80)
    ap.add_argument('--max_episode', type=int, default=500,
                    help='pre-registered: the detector must fire at or before this episode')
    ap.add_argument('--cell', default=None,
                    help='CONFIRMATORY mode: test exactly ONE pre-specified cell, given as '
                         '"observable:lo-hi:feature" e.g. "pt_mean:0-300:std". Reports a '
                         'SINGLE-TEST permutation p with NO family-wise correction, because only '
                         'one hypothesis is tested. The grid is still printed but is labelled '
                         'exploratory and does not enter the verdict.')
    ap.add_argument('--out', default='basin_detect')
    a = ap.parse_args()

    runs = load(a.dir, a.threshold)
    if not runs:
        print(f"no runs found under {a.dir}"); sys.exit(1)
    idx = list(enumerate(runs))
    n_lo = sum(r['low'] for r in runs)
    print(f"loaded {len(runs)} runs | {n_lo} low-power / {len(runs) - n_lo} high-power "
          f"| threshold {a.threshold} dBm")
    cfgs = sorted({r['config'] for r in runs})
    print("  per config:")
    for c in cfgs:
        rr = [r for r in runs if r['config'] == c]
        print(f"    {c:<16} {sum(x['low'] for x in rr):>2} low / {len(rr):>2} runs")
    print(f"\nPRE-REGISTERED: within-config AUC >= {a.auc_target:.2f} at a window ending "
          f"<= ep {a.max_episode}, family-wise p < 0.05.\n")

    windows = [(0, int(w)) for w in a.windows.split(',')]
    observables = [o for o in ['return', 'pt_mean', 'pt_last', 'rb_mean', 'ho']
                   if any(o in r['series'] for r in runs)]
    print(f"  observables logged: {', '.join(observables)}")
    if 'pt_mean' not in observables:
        print("  NOTE: no pt_mean column -- this sweep predates the actuator-trace patch, so")
        print("        only return is testable. P4 cannot be evaluated on these runs.")
    print()

    X = precompute(runs, windows, observables)
    lab = np.array([r['low'] for r in runs], bool)
    cfg_groups = [np.array([i for i, r in enumerate(runs) if r['config'] == c]) for c in cfgs]

    print(f"  {'observable':<11}{'window':<13}{'feature':<20}{'within-cfg AUC':>16}{'pooled':>9}")
    print("  " + "-" * 69)
    cells, best = [], None
    for obs in observables:
        for lo, hi in windows:
            for f in FEATURES:
                col = X[(obs, lo, hi, f)]
                wa = wc_auc(col, lab, cfg_groups)
                ok = np.isfinite(col)
                pa = auc(col[ok], lab[ok]) if ok.sum() else np.nan
                if not np.isfinite(wa):
                    continue
                cells.append(dict(obs=obs, lo=lo, hi=hi, feat=f,
                                  auc_within=float(wa), auc_pooled=float(pa)))
                print(f"  {obs:<11}ep {lo}-{hi:<8}{f:<20}{wa:>16.3f}{pa:>9.3f}")
                if hi <= a.max_episode and (best is None or
                                            abs(wa - .5) > abs(best['auc_within'] - .5)):
                    best = cells[-1]
        print()

    if not cells:
        print("\n" + "=" * 72)
        print("  CANNOT ADJUDICATE -- no cell had at least 3 runs in each class.")
        nlo = int(lab.sum())
        print(f"  This sweep produced {nlo} low-power and {len(lab) - nlo} high-power runs.")
        print("  With one config the >=3-per-class guard needs at least 3 of each.")
        print("  NOTE: an extreme split is ITSELF a result -- it falsifies P3 (the ~6/20")
        print("  prediction) even though the detection test cannot run. Report it as such.")
        sys.exit(0)

    if a.cell:
        try:
            c_obs, c_win, c_feat = a.cell.split(":")
            c_lo, c_hi = (int(v) for v in c_win.split("-"))
        except Exception:
            print(f'  --cell must look like "pt_mean:0-300:std", got {a.cell!r}'); sys.exit(1)
        key = (c_obs, c_lo, c_hi, c_feat)
        if key not in X:
            print(f"  cell {a.cell} not available. observables={observables} "
                  f"windows={[w[1] for w in windows]} features={FEATURES}")
            sys.exit(1)
        col = X[key]
        obs_auc = wc_auc(col, lab, cfg_groups)
        if not np.isfinite(obs_auc):
            print(f"  cell {a.cell} could not be evaluated (class balance)."); sys.exit(0)
        rng1 = np.random.default_rng(0); hits1 = 0
        for _ in range(a.iters):
            perm = lab.copy()
            for g in cfg_groups:
                perm[g] = rng1.permutation(lab[g])
            v = wc_auc(col, perm, cfg_groups)
            if np.isfinite(v) and abs(v - 0.5) >= abs(obs_auc - 0.5) - 1e-12:
                hits1 += 1
        p1 = (hits1 + 1) / (a.iters + 1)
        lo_m = float(np.nanmean(col[lab])); hi_m = float(np.nanmean(col[~lab]))
        direction = "HIGHER" if lo_m > hi_m else "LOWER"
        print("\n" + "=" * 72)
        print("CONFIRMATORY VERDICT -- one pre-specified cell, single-test p, no correction")
        print("=" * 72)
        print(f"  cell           : {a.cell}")
        print(f"  AUC            : {obs_auc:.3f}   (pre-registered bar {a.auc_target:.2f})")
        print(f"  single-test p  : {p1:.4f}   ({a.iters} permutations, NO family-wise correction)")
        print(f"  direction      : low basin {lo_m:.4f} vs high {hi_m:.4f} -> low is {direction}")
        print(f"  sample         : {int(lab.sum())} low / {len(lab) - int(lab.sum())} high")
        passed = bool(obs_auc >= a.auc_target and p1 < 0.05)
        print()
        if passed:
            print("  CONFIRMED on fresh seeds. The pre-registered hypothesis holds out of sample.")
            print("  Early actuator volatility predicts the basin. Run the PRIOR-ART GATE before")
            print("  designing any steering mechanism.")
        else:
            print("  NOT CONFIRMED. The exploratory finding did not replicate on fresh seeds.")
            print("  Early detection is dead on BOTH observables. The steering line closes, and the")
            print("  honest output is the negative result: a 9x reliability lever that is real,")
            print("  consequential, deterministic given the seed, and not predictable from any")
            print("  observable available during training.")
        print("\n  The grid above is EXPLORATORY and did not enter this verdict.")
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "basin_confirmatory.json"), "w") as f:
            json.dump(dict(cell=a.cell, auc=float(obs_auc), p_single=float(p1), low_mean=lo_m,
                           high_mean=hi_m, passed=passed, n=len(lab), n_low=int(lab.sum())), f, indent=1)
        print(f"  wrote {os.path.join(a.out, 'basin_confirmatory.json')}")
        return

    print(f"  family-wise permutation ({a.iters} iters, labels shuffled within config,")
    print(f"  max statistic over all {len(cells)} cells across every observable)...")
    obs_max = max(abs(c['auc_within'] - 0.5) for c in cells)
    cols = [X[(c['obs'], c['lo'], c['hi'], c['feat'])] for c in cells]
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(a.iters):
        perm = lab.copy()
        for g in cfg_groups:
            perm[g] = rng.permutation(lab[g])
        m = 0.0
        for col in cols:
            v = wc_auc(col, perm, cfg_groups)
            if np.isfinite(v):
                m = max(m, abs(v - 0.5))
                if m >= obs_max - 1e-12:
                    break
        if m >= obs_max - 1e-12:
            hits += 1
    p_fwe = (hits + 1) / (a.iters + 1)

    print("\n" + "=" * 72)
    print("VERDICT (pre-registered rule, applied as written)")
    print("=" * 72)
    desc = ('%s  ep %d-%d  %s  AUC %.3f' % (best['obs'], best['lo'], best['hi'],
                                            best['feat'], best['auc_within'])) if best else 'none'
    print(f"  strongest early cell (ending <= ep {a.max_episode}): {desc}")
    print(f"  family-wise p over {len(cells)} cells: {p_fwe:.4f}\n")
    fires = best is not None and abs(best['auc_within'] - 0.5) + 0.5 >= a.auc_target and p_fwe < 0.05
    if fires:
        print("  A DETECTOR IS PLAUSIBLE. Early return separates the basins before one third")
        print("  of training. Run the PRIOR-ART GATE before designing anything -- mode collapse,")
        print("  lottery tickets, plasticity and AutoML run-triage are all plausible occupants.")
        if best['auc_within'] > 0.95 or best['auc_within'] < 0.05:
            print("\n  BUT READ THIS BEFORE CELEBRATING. An AUC this close to perfect means")
            print("  detection is TRIVIAL, and trivial detection is not a contribution. The")
            print("  low-power basin earns MORE return (89.4 vs 80.3) because it pays less")
            print("  power penalty while failing more often -- so a near-perfect AUC may just")
            print("  be re-reading the objective the run is already optimising, arriving early.")
            print("  If detection is this easy, the contribution is NOT the detector. It is")
            print("  STEERING: intervening on a flagged run so it converges into the reliable")
            print("  basin instead. Frame it that way, and make the detector a one-line")
            print("  baseline rather than the result.")
    else:
        print("  NO EARLY DETECTOR IN RETURN. Per-episode return does not carry basin")
        print("  membership at the pre-registered bar. This does not kill the basin lever --")
        print("  it kills THIS observable. The next cheapest step is a one-line change to log")
        print("  mean transmit power per episode, then retrain a sweep on FIST and re-run this")
        print("  script against that. Power is the quantity that DEFINES the basin, so it is")
        print("  the natural next observable and it was simply never recorded per episode.")

    pooled_gap = np.nanmean([abs(c['auc_pooled'] - c['auc_within']) for c in cells])
    print(f"\n  confound check: mean |pooled - within-config| AUC = {pooled_gap:.3f}")
    if pooled_gap > 0.05:
        print("  Pooling across configs shifts the AUC materially -- configs differ in basin")
        print("  propensity AND in return, exactly the confound this design controls for.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'basin_early_detect.json'), 'w') as f:
        json.dump(dict(cells=cells, p_fwe=p_fwe, best=best, n=len(runs), n_low=int(n_lo)), f, indent=1)
    print(f"\n  wrote {os.path.join(a.out, 'basin_early_detect.json')}")


if __name__ == '__main__':
    main()
