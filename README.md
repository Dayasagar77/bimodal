# Bimodal convergence — analysis, pre-registrations, and every run behind the paper

Analysis code and complete run record for *"Bimodal Convergence in Deep Reinforcement Learning
for Sub-THz Vehicular Radio Resource Management: A Pre-Registered Replication."*
Daya Sagar G and Deepa Nivethika S., School of Computer Science and Engineering,
Vellore Institute of Technology, Chennai.

**You do not need to retrain anything to check this paper.** Every number in it is derived from
the 100 run records in `runs/`, which are included. A full replication from scratch is possible
too — the environment and controller are public at
[Dayasagar77/Anticipate](https://github.com/Dayasagar77/Anticipate) — but nothing here requires it.

## What the paper reports

Training a branched QR-DQN controller on real DAIR-V2X vehicular trajectories at 140 GHz
converges to one of **exactly two** policies: transmit power near the 30 dBm ceiling, or near
the 23 dBm floor.

- Across **80 runs spanning a twenty-two-fold change** in the reward's transmit-power penalty,
  the interval between the two is never narrower than **5.744 dB**, and **not one run falls
  inside it**.
- Which basin a run reaches decides reliability: sustained radio-link failure of
  **8.79% ± 2.65** against **1.05% ± 0.74**, with **all 1116** low-versus-high pairwise
  comparisons favouring the high-power basin (permutation *p* < 5 × 10⁻⁶).
- The structure was found post hoc, then **pre-registered and confirmed** on twenty seeds never
  previously trained.
- It is **deterministic given the seed and not predictable from any observable available during
  training.**

## Two pre-registered predictions failed, and are reported as failures

`prereg/` carries both pre-registrations verbatim, with their original dates, written before the
runs they govern existed.

- **P4** predicted an early-detection AUC ≥ 0.80 from per-episode transmit-power volatility.
  On fresh seeds it reached **0.653** (*p* = 0.347), after an exploratory search of 150
  statistics had suggested 0.869. That is the winner's curse, caught by the design.
- **W2** predicted the low-basin count would rise monotonically with the reward weight. Observed
  3, 2, 6, 7 — inverted between the two cheapest arms.

Neither was quietly dropped. Both are in the paper.

## Layout

| | |
|---|---|
| `prereg/` | the two pre-registrations, unedited |
| `analysis/basin_early_detect.py` | within-configuration AUC, max-statistic permutation, the confirmatory single-cell test |
| `analysis/wpwr_sweep.py` | the four-arm reward-weight sweep driver, with the three no-op guards |
| `analysis/wpwr_adjudicate.py` | scores the sweep against the pre-registration |
| `analysis/wpwr_override_proof.py` | proves the weight override is exact to 1.9 × 10⁻¹⁵ and touches nothing but the reward |
| `analysis/patch_wpwr_20260830.py` | the idempotent patch that adds the override to the environment |
| `analysis/make_figures.py` | the three figures |
| `runs/basin_v2` | 20 runs, W_pwr = 0.15, seeds 0–19 — the exploratory set |
| `runs/basin_v3` | 20 runs, W_pwr = 0.15, seeds 20–39 — **the pre-registered replication** |
| `runs/wpwr_w0.02`, `w0.05`, `w0.45` | 20 runs each, the reward-weight arms |
| `runs/wpwr_adjudication_runs.csv` | all 80 weight-sweep runs in one table |

Each run directory holds, per seed: `_metrics.csv` (the headline numbers), `_returns.csv`
(per-episode return and actuator trace) and `_eval.csv` (per-episode held-out results).

## Checking a claim

```bash
python analysis/wpwr_adjudicate.py     # scores every W1-W4 prediction against runs/
python analysis/basin_early_detect.py --dir runs/basin_v3 --cell "pt_mean:0-300:std"
python analysis/make_figures.py
```

## Data

The run records are **our own training outcomes** — no DAIR-V2X data is redistributed here.
Rebuilding the runs from scratch needs the environment from
[Anticipate](https://github.com/Dayasagar77/Anticipate) and DAIR-V2X from its own authors under
their licence. Trained checkpoints are omitted for size and are available from the corresponding
author.

## Licence

MIT — see [LICENSE](LICENSE).
