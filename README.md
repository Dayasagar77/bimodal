# Bimodal convergence — analysis, pre-registrations, and every run behind the paper

Analysis code and complete run record for *"Bimodal Convergence in Deep Reinforcement Learning
for Sub-THz Vehicular Radio Resource Management: A Pre-Registered Replication."*
Daya Sagar G and Deepa Nivethika S., School of Computer Science and Engineering,
Vellore Institute of Technology, Chennai.

**You do not need to retrain anything to check this paper.** Every number in it is derived from
the 180 run records in `runs/`, which are included. A full replication from scratch is possible
too — the environment and controller are public at
[Dayasagar77/Anticipate](https://github.com/Dayasagar77/Anticipate) — but nothing here requires it.

## What the paper reports

Training a branched QR-DQN controller on real DAIR-V2X vehicular trajectories at 140 GHz
converges to one of **exactly two** policies: transmit power near the 30 dBm ceiling, or near
the 23 dBm floor.

- Across **80 runs spanning a twenty-two-fold change** in the reward's transmit-power penalty,
  the interval between the two is never narrower than **5.744 dB**, and **not one run falls
  inside it**.
- Changing the penalty's **functional form** — pricing intermediate power quadratically cheaper,
  or square-root dearer, while holding both rail policies at identical cost — does not populate
  the interval either. Of the **120 runs** in which the power objective was manipulated, the one
  whose mean lands inside turns out to alternate between the two rails across episodes rather
  than to hold an intermediate power.
- **Two published explanations are tested and neither survives.** The risk-averse
  local-optimum barrier (Greenberg et al., NeurIPS 2022) is refuted: removing CVaR *doubles* the
  split, 6/20 to 13/20. Linear scalarisation's non-convexity (Vamplew et al., 2008) is
  unsupported: pricing the interior cheaper yielded 0 of 20 interior runs.
- **A third explanation is CONFIRMED by controlled manipulation: the actuator.** Three power
  actuators over the *identical* reachable set {23..30} dBm, differing only in how many steps it
  takes to cross it, give 0, 0 and 7 committed interior policies out of 20 for `inc3` (3 actions,
  7 steps), `inc7` (7 actions, 3 steps) and `direct8` (8 actions, 1 step). `inc7` is the control:
  more than twice the branch width and less than half the traversal cost, and it leaves the
  interval empty. **The empty interval is a property of incremental action ADDRESSING** - not the
  reward, the risk measure, the branch width or the traversal speed.
- Which basin a run reaches decides reliability: sustained radio-link failure of
  **8.79% ± 2.65** against **1.05% ± 0.74**, with **all 1116** low-versus-high pairwise
  comparisons favouring the high-power basin (permutation *p* < 5 × 10⁻⁶).
- The structure was found post hoc, then **pre-registered and confirmed** on twenty seeds never
  previously trained.
- It is **deterministic given the seed and not predictable from any observable available during
  training.**

## Eight pre-registered predictions failed, and all eight are reported as failures

`prereg/` carries all three pre-registrations verbatim, with their original dates, written before
the runs they govern existed.

- **P4** predicted an early-detection AUC ≥ 0.80 from per-episode transmit-power volatility.
  On fresh seeds it reached **0.653** (*p* = 0.347), after an exploratory search of 150
  statistics had suggested 0.869. That is the winner's curse, caught by the design.
- **W2** predicted the low-basin count would rise monotonically with the reward weight. Observed
  3, 2, 6, 7 — inverted between the two cheapest arms.

- **Q1-Q4** governed the power-penalty form sweep, and all four failed. Q1 predicted the
  quadratic form would put at least one run inside the interval; it put none, and halved the low
  basin instead. Q2 predicted the interval would stay empty under the square-root form; one run's
  mean landed inside — that run is the within-run mixture described above. Q3 held in both arms
  the pre-registration made eligible, and failed only on an arm its own gate excludes. Q4 bounded
  the rail locations, and **both new arms breached it by 0.022 dB and 0.126 dB** against a 5.9 dB
  gap — bounds we set too tightly, not rails that moved.

None of these was quietly dropped. All are in the paper. On Q4 we did not waive our own kill rule
to keep the result: **both form arms are reported as exploratory rather than confirmatory**, and
the breach is stated in the manuscript.

## Layout

| | |
|---|---|
| `prereg/` | the four pre-registrations, unedited |
| `analysis/basin_early_detect.py` | within-configuration AUC, max-statistic permutation, the confirmatory single-cell test |
| `analysis/wpwr_sweep.py` | the four-arm reward-weight sweep driver, with the three no-op guards |
| `analysis/wpwr_adjudicate.py` | scores the sweep against the pre-registration |
| `analysis/wpwr_override_proof.py` | proves the weight override is exact to 1.9 × 10⁻¹⁵ and touches nothing but the reward |
| `analysis/patch_wpwr_20260830.py` | the idempotent patch that adds the override to the environment |
| `analysis/pwrform_sweep.py` | the power-penalty form sweep driver, with the same guards |
| `analysis/pwrform_proof.py` | proves every form prices the two rails identically and only the interior differently |
| `analysis/pwrform_reward_proof.py` | the per-step reward check for the form override |
| `analysis/patch_pwrform_20260830.py` | the idempotent patch that adds the form override |
| `analysis/make_figures.py` | figures 1-3 |
| `analysis/pwract_sweep.py` | the actuator sweep driver |
| `analysis/pwract_proof.py` | proves all three actuators reach the identical power set, and that the published one is reproduced exactly |
| `analysis/pwract_adjudicate.py` | scores A1-A4 against the runs |
| `analysis/patch_pwract_20260830.py` | the idempotent patch adding the selectable actuator |
| `analysis/make_fig4.py` | figure 4 — the within-run mixture, and the 120-run corpus view |
| `analysis/make_fig5.py` | figure 5 — where each actuator's twenty runs converged |
| `runs/basin_v2` | 20 runs, W_pwr = 0.15, seeds 0–19 — the exploratory set |
| `runs/basin_v3` | 20 runs, W_pwr = 0.15, seeds 20–39 — **the pre-registered replication** |
| `runs/wpwr_w0.02`, `w0.05`, `w0.45` | 20 runs each, the reward-weight arms |
| `runs/pwrform_quad`, `pwrform_sqrt` | 20 runs each, the power-penalty form arms |
| `runs/pwract_direct8`, `pwract_inc7` | 20 runs each, the actuator arms |
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
