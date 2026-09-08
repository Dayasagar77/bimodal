"""Re-derive every cell of the reference-policy table in SIII from the run data.

Reads baselines/out500/baselines_metrics.csv, which paper2_baselines.py wrote when the
non-learned rules were evaluated on the same 500 held-out episodes and the same reset
seeds (CRN base 900,000,000) as every learned run in this paper.
"""
import pandas as pd, re, sys

d = pd.read_csv('baselines/out500/baselines_metrics.csv').set_index('config')
md = open('paper3_basins.md', encoding='utf8').read()
flat = re.sub(r'\s+', ' ', md)   # the manuscript is hard-wrapped; match on flattened text

ROWS = {
    'random':       ('random',       12.40, 27.23, 75.49, 23.16),
    'static_hold':  ('static, hold', 14.60, 30.00, 69.34,  0.00),
    'static_low':   ('static, low',  28.80, 23.37, 74.41,  0.00),
    'static_eff':   ('static, eff',  15.80, 25.84, 81.36,  0.00),
    'maxsinr_hold': ('max-SINR, hold', 10.40, 30.00, 68.06, 25.69),
    'maxsinr_low':  ('max-SINR, low',  15.60, 23.34, 77.09, 24.92),
    'maxsinr_eff':  ('max-SINR, eff',  12.40, 25.43, 80.31, 25.48),
    'a3_hold':      ('A3, hold',     10.60, 30.00, 70.26,  3.84),
    'a3_low':       ('A3, low',      16.80, 23.36, 79.04,  3.67),
    'a3_eff':       ('A3, eff',      13.20, 25.16, 82.56,  3.77),
    'genie_hold':   ('genie, hold',   1.60, 30.00, 77.55,  2.42),
    'genie_eff':    ('genie, eff',    2.00, 24.80, 93.01,  2.42),
}

ok = fail = 0
def chk(label, got, want, tol):
    global ok, fail
    if abs(got - want) <= tol:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL {label}: manuscript {want}, data {got:.4f}")

for cfg, (label, rlf, pt, ret, ho) in ROWS.items():
    assert cfg in d.index, f'{cfg} missing from the metrics file'
    r = d.loc[cfg]
    assert int(r['n_eval']) == 500, f'{cfg}: n_eval is {r["n_eval"]}, not 500'
    chk(f'{label} RLF',        float(r['rlf_rate']),    rlf, 5e-3)
    chk(f'{label} pt',         float(r['pt_dbm']),      pt,  5e-3)
    chk(f'{label} return',     float(r['return_mean']), ret, 5e-3)
    chk(f'{label} handovers',  float(r['ho_per_ep']),   ho,  5e-3)
    # the printed row must actually be in the manuscript
    assert f'| {label} |' in flat or f'| **{label}** |' in flat or f'| *{label} (upper bound)* |' in flat, \
        f'row for {label} not found in the manuscript'

# the two derived claims in the prose
best_conv = min(float(d.loc[c, 'rlf_rate']) for c in
                ['random','static_hold','static_low','static_eff','maxsinr_hold','maxsinr_low',
                 'maxsinr_eff','a3_hold','a3_low','a3_eff'])
chk('best deployable non-learned RLF', best_conv, 10.40, 5e-3)
chk('static hold-vs-low power cost (pp)',
    float(d.loc['static_low','rlf_rate']) - float(d.loc['static_hold','rlf_rate']), 14.2, 5e-3)
chk('A3 hold-vs-low power cost (pp)',
    float(d.loc['a3_low','rlf_rate']) - float(d.loc['a3_hold','rlf_rate']), 6.2, 5e-3)
assert 'costs the static rule 14.2 pp and the A3 rule 6.2 pp' in flat, 'prose figures changed'
assert '9.82 % for the low regime against 10.40 % for the best conventional rule' in flat, 'scale claim changed'

print(f'{ok} checks passed, {fail} failed')
sys.exit(1 if fail else 0)
