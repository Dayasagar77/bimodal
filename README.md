# Bimodal convergence — analysis, pre-registrations and run records

Analysis code, pre-registrations and run records for the bimodal-convergence experiments of:

> **Bimodal Power Convergence Turns Deep Reinforcement Learning for 140 GHz Vehicular
> Networks into a Reliability Lottery**
> Daya Sagar G and Deepa Nivethika S., School of Computer Science and Engineering,
> Vellore Institute of Technology, Chennai.

The paper's pinning experiments are archived at
[Dayasagar77/branch-lottery](https://github.com/Dayasagar77/branch-lottery); the rest of its
material is listed in its Data and Code Availability section.

**You do not need to retrain anything to check these experiments.** Their run records are in
`runs/` and are included. A full replication from scratch is possible too — the environment and
controller are public at [Dayasagar77/Anticipate](https://github.com/Dayasagar77/Anticipate) — but
nothing here requires it.

## What these runs show

A branched QR-DQN controller trained on real DAIR-V2X vehicular trajectories at 140 GHz converges,
run by run, to one of two regimes: transmit power near the 30 dBm ceiling, or near the 23 dBm
floor.

- **Found after the fact, then confirmed.** The split was first seen on sweep A (seeds 0–19).
  Before sweep B existed, its structure was pre-registered; on those twenty fresh seeds (20–39)
  it returned an empty interval of 5.89 dB, no run between 25 and 28 dBm, a reliability
  difference of 9.83 pp and 5 low runs of 20.
- **The regime decides reliability.** In sweeps A and B together, low-regime runs fail in
  **9.82% ± 2.67%** of held-out episodes and high-regime runs in **1.03% ± 0.93%**, and every one
  of the 319 low-versus-high pairs has the low run worse (exact one-sided permutation
  *p* = 6.45 × 10⁻⁵ for the confirmatory sweep B).
- **The reward sets how often the floor is reached, not whether.** Across **80 runs spanning a
  22-fold change** in the weight of the transmit-power penalty, the empty interval never narrows
  below **5.74 dB** and **no run lands between 25 and 28 dBm**. Repricing intermediate power,
  cheaper (*u*²) or dearer (√*u*) with the floor and the ceiling unchanged, does not fill it
  either: the one run whose mean lands inside (√*u*, seed 16) is a rail mixture, at the floor in
  some scenes and at the ceiling in others. Linear scalarisation (Vamplew et al., 2008) does not
  explain the empty interval.
- **Removing risk aversion does not remove the split.** In the controller-variant sweep, removing
  CVaR raised the low regime from 6 to 13 runs of 20 (exploratory, on shared seeds; two-sided
  Fisher's exact *p* = 0.056), which runs against a local-optimum barrier created by risk
  aversion (Greenberg et al., NeurIPS 2022).
- **The power actuator shapes the split.** Three actuators over the *identical* reachable set
  {23..30} dBm give **0, 0 and 7** committed interior policies out of 20: `inc3` (the baseline;
  relative, 7 steps to cross), `inc7` (the control; relative, 3 steps, more than twice the branch
  width) and `direct8` (absolute, 1 step); Fisher's exact one-sided *p* = 4.2 × 10⁻³ for
  `direct8` against `inc7`. The empty interval is therefore associated with relative addressing,
  not with branch width or traversal speed. Absolute addressing also changes exploration, credit
  assignment and the geometry of the values over the branch, which the control does not separate.
  It changes the shape of the outcome, not its cost: the `direct8` arm fails in 3.22% of episodes
  against 3.34% for `inc3`.
- **A policy-gradient learner splits too.** Under a pre-registration, 20 PPO runs reproduce the
  empty interval (4.74 dB, no run between 25 and 28 dBm), with a reliability difference of
  3.48 pp, short of the 5 pp predicted; the arm passed its competence gate only narrowly.
- **Deterministic, and not visible early.** Training is deterministic given the seed: re-running
  seeds 0–19 reproduced all 20 runs. A power-volatility statistic that an exploratory search over
  150 statistics had put at AUC 0.869 did not replicate on fresh seeds under pre-registration
  (AUC 0.653, *p* = 0.347), and none of the signals tested supports stopping runs early.

## Eight pre-registered predictions failed, one could not be evaluated, and every one is reported

`prereg/` carries the five pre-registrations behind these runs verbatim, with their original
dates, each written before the runs it governs existed. The paper's table of pre-registered
predictions and outcomes gives the complete record, together with those of its other experiments.

- **P4** predicted an early-detection AUC ≥ 0.80 from per-episode transmit-power volatility.
  On fresh seeds it reached **0.653** (*p* = 0.347), after an exploratory search of 150
  statistics had suggested 0.869. The exploratory value did not replicate; with five low-regime
  runs in sweep B, the test had little power.
- **W2** predicted the low-regime count would rise monotonically with the reward weight. Observed
  3, 2, 6, 7 — inverted between the two cheapest arms.

- **Q1, Q2 and Q4** governed the penalty-shape sweep. Q1 predicted the quadratic form would put at
  least one run inside the interval; it put none, and cut the low regime from 6 to 2 of 20
  instead. Q2 predicted the interval would stay empty under the square-root form; one run's mean
  landed inside — that run is the rail mixture described above — while its count prediction held
  (7 of 20). Q4 bounded the rail locations, and **both new arms breached it by 0.022 dB and
  0.126 dB** against a 5.9 dB gap — bounds we set too tightly, not rails that moved. **Q3 held**
  in both arms the pre-registration made eligible; it does not apply to the arm its own gate
  excludes.
- **A3** governed the actuator sweep and failed on a criterion of our own construction: it required
  interior reliability to fall between the low- and high-regime rates *of the same arm*, but every
  committed interior policy under direct addressing is itself labelled low-regime, so the test is
  circular exactly where it matters. **A4** is the unevaluable one — it held in `inc7` (0.119 dB
  against a 0.25 dB tolerance) and has no referent in `direct8`, which the manipulation left with no
  rails to locate. A1 and A2, the decisive and attribution rules, both held.
- **X2 and X3** governed the PPO arm. The reliability difference was 3.48 pp against the 5 pp
  predicted, and two high runs, at 28.09 and 28.51 dBm, broke the threshold invariance. X1 and the
  competence gate held.

None of these was quietly dropped. All are in the paper. On Q4 we did not waive our own kill rule
to keep the result: **both shape arms are reported as exploratory rather than confirmatory**, and
the breach is stated in the paper.

## Layout

| | |
|---|---|
| `prereg/` | the five pre-registrations, unedited, with an index (`prereg/README.md`) and the PPO outcome |
| `analysis/basin_early_detect.py` | within-configuration AUC, max-statistic permutation, the confirmatory single-cell test |
| `analysis/wpwr_sweep.py` | the four-arm reward-weight sweep driver, with the three no-op guards |
| `analysis/wpwr_adjudicate.py` | scores the sweep against the pre-registration |
| `analysis/wpwr_override_proof.py` | proves the weight override is exact to 1.9 × 10⁻¹⁵ and touches nothing but the reward |
| `analysis/patch_wpwr_20260830.py` | the idempotent patch that adds the override to the environment |
| `analysis/pwrform_sweep.py` | the power-penalty form sweep driver, with the same guards |
| `analysis/pwrform_proof.py` | proves every form prices the two rails identically and only the interior differently |
| `analysis/pwrform_reward_proof.py` | the per-step reward check for the form override |
| `analysis/patch_pwrform_20260830.py` | the idempotent patch that adds the form override |
| `analysis/make_figures.py` | first version of the reward-weight, reliability and two-lotteries figures |
| `analysis/pwract_sweep.py` | the actuator sweep driver |
| `analysis/pwract_proof.py` | proves all three actuators reach the identical power set, and that the original actuator (`inc3`) is reproduced exactly |
| `analysis/pwract_adjudicate.py` | scores A1-A4 against the runs |
| `analysis/patch_pwract_20260830.py` | the idempotent patch adding the selectable actuator |
| `analysis/make_fig4.py` | first version of the within-run mixture figure and the 120-run corpus view |
| `analysis/make_fig5.py` | first version of the actuator figure: where each actuator's twenty runs converged |
| `analysis/make_figures_v2.py` | the revised figure set, whose output is the six PDFs in `figures/`; `basin_v2_runs.csv` and `fig5_actuators.csv` are per-run inputs to it |
| `analysis/ppo_arm.py`, `adjudicate_ppo.py`, `crosscheck_ppo.py` | the PPO arm: training and evaluation, the registered rules, and the check of every PPO number (see `README_PPO.md`) |
| `analysis/paper2_baselines.py` | the non-learned reference policies, evaluated on the same 500 held-out episodes and reset seeds as every learned run |
| `analysis/crosscheck_baselines.py` | re-derives every reference-policy number from the run data and checks it against the manuscript text |
| `runs/basin_v2` | sweep A: 20 runs, W_pwr = 0.15, seeds 0–19 — the exploratory set |
| `runs/basin_v3` | sweep B: 20 runs, W_pwr = 0.15, seeds 20–39 — **the pre-registered replication** |
| `runs/wpwr_w0.02`, `w0.05`, `w0.45` | 20 runs each, the reward-weight arms |
| `runs/pwrform_quad`, `pwrform_sqrt` | 20 runs each, the penalty-shape arms (*u*², √*u*) |
| `runs/pwract_direct8`, `pwract_inc7` | 20 runs each, the actuator arms |
| `runs/ppo` | 20 PPO runs, seeds 0–19, with the adjudication output |
| `runs/baselines` | the reference policies — random, and static, max-SINR, A3 and genie handover, each at three power settings (hold, low, eff): per-episode held-out results and a summary. The time-to-blockage rule is not here |
| `runs/wpwr_adjudication_runs.csv` | all 80 weight-sweep runs in one table |
| `runs/sixconfig/` | the controller-variant sweep — six configurations, 120 runs, `all_metrics.csv` plus one `_returns.csv` per run. **Not among the run records above**: these runs were produced for the ablation study of the companion manuscript that introduced the controller and are held with that project; what is here is the summary and the per-episode return series from which the paper's controller-variant results are computed |
| `analysis/sixconfig_export.py` | regenerates `runs/sixconfig/` from the source sweep and refuses to write unless it reproduces Table IV of the companion manuscript, the per-seed basin counts and the chi-square test |
| `ENVIRONMENT.md` | interpreter and package versions, and every call site at which the run seed enters the pipeline |
| `README_PPO.md` | the PPO arm's files, and how to reproduce one seed |

Each run directory holds, per seed: `_metrics.csv` (the headline numbers), `_returns.csv`
(per-episode return and actuator trace) and `_eval.csv` (per-episode held-out results);
`runs/ppo` has `_train.csv` in place of `_returns.csv`.

## Checking a claim

```bash
python analysis/wpwr_adjudicate.py     # scores every W1-W4 prediction against runs/
python analysis/basin_early_detect.py --dir runs/basin_v3 --cell "pt_mean:0-300:std"
python analysis/sixconfig_export.py --src <sweep_r3>   # checks the companion manuscript's Table IV, then writes runs/sixconfig
python analysis/basin_early_detect.py --dir runs/sixconfig --iters 20000   # the within-configuration AUCs of the controller-variant sweep
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
