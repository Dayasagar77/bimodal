"""Re-derive every PPO cell in SIV-J from runs/ppo/*_metrics.csv."""
import glob, re, sys
import numpy as np, pandas as pd

D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('ppo/runs/ppo_s*_metrics.csv'))],
              ignore_index=True).sort_values('seed')
assert len(D) == 20, f'expected 20 PPO runs, got {len(D)}'
D['low'] = D.pt_dbm < 27.0
lo, hi = D[D.low], D[~D.low]
md = open('paper3_basins.md', encoding='utf8').read()
flat = re.sub(r'\s+', ' ', md)

ok = fail = 0
def chk(lbl, got, want, tol):
    global ok, fail
    if abs(got - want) <= tol: ok += 1
    else: fail += 1; print(f'  FAIL {lbl}: manuscript {want}, data {got:.4f}')
def instr(s):
    global ok, fail
    if s in flat: ok += 1
    else: fail += 1; print(f'  FAIL not in manuscript: {s!r}')

chk('low count', len(lo), 12, 0);            chk('high count', len(hi), 8, 0)
chk('low min',  lo.pt_dbm.min(), 23.248, 5e-4)
chk('low max',  lo.pt_dbm.max(), 23.346, 5e-4)
chk('high min', hi.pt_dbm.min(), 28.087, 5e-4)
chk('high max', hi.pt_dbm.max(), 30.000, 5e-4)
chk('empty interval', hi.pt_dbm.min() - lo.pt_dbm.max(), 4.741, 5e-4)
chk('runs inside 25-28', int(((D.pt_dbm > 25) & (D.pt_dbm < 28)).sum()), 0, 0)
chk('RLF low',  lo.rlf_rate.mean(), 13.18, 5e-3)
chk('RLF high', hi.rlf_rate.mean(),  9.70, 5e-3)
chk('RLF diff pp', lo.rlf_rate.mean() - hi.rlf_rate.mean(), 3.48, 5e-3)
chk('corr(RLF, pt)', np.corrcoef(D.rlf_rate, D.pt_dbm)[0, 1], -0.440, 5e-4)
chk('median RLF', D.rlf_rate.median(), 12.00, 5e-3)
chk('best run RLF', D.rlf_rate.min(), 0.80, 5e-3)
chk('second best RLF', np.sort(D.rlf_rate.values)[1], 1.80, 5e-3)
chk('high-regime runs at 10.60', int((hi.rlf_rate == 10.60).sum()), 5, 0)
chk('rank separation incomplete', int(lo.rlf_rate.min() > hi.rlf_rate.max()), 0, 0)
chk('handover-high runs', int((D.ho_per_ep > 10).sum()), 6, 0)
chk('handover-low runs',  int((D.ho_per_ep < 1).sum()), 13, 0)
chk('handover between',  int(((D.ho_per_ep >= 1) & (D.ho_per_ep <= 10)).sum()), 1, 0)

for s in ['12 / 20 at 23.248–23.346 dBm', '8 / 20 at 28.087–30.000 dBm', '**4.741 dB**',
          '13.18 % vs 9.70 % (+3.48 pp)', '−0.440', '12.00 %',
          '(0.80 % and 1.80 % RLF) and five at 10.60 %',
          'six runs settle above ten handovers per episode and thirteen below one']:
    instr(s)

print(f'{ok} checks passed, {fail} failed')
sys.exit(1 if fail else 0)
