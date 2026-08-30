# PRE-REGISTRATION — reward-weight (W_PWR) sensitivity of the bimodal convergence structure

**Written 30 August 2026, BEFORE any arm was trained.** Every number below is derived from
`runs/basin_v2` (seeds 0-19, W_PWR = 0.15), which already existed. No new arm had been run at
the time of writing.

## 1. Why this exists

The reward is

```
r = W_rel*1{link_ok} - W_rlf*1{RLF} - W_pwr*(pt-PT_MIN)/(PT_MAX-PT_MIN)
    - W_spec*(rb-RB_MIN)/(RB_MAX-RB_MIN) - W_ho*1{ho}
(W_rel, W_rlf, W_pwr, W_spec, W_ho) = (1.0, 20.0, 0.15, 0.05, 0.05)
PT_MIN, PT_MAX = 23., 30.      SINR_out = -13.8 dB
```

At the published W_PWR = 0.15 the low-power basin earns the **higher** mean return (89.38 vs
80.27). The objective as posed prefers the nine-times-less-reliable policy. The objection this
invites: *"the bimodality is an artefact of a linear power penalty against a step-shaped
reliability cliff - retune W_PWR and it disappears."*

The existing six-configuration sweep does not answer it. All six are ablations of the
**controller** and every one shares identical reward weights. The reward-weight axis has never
been varied.

## 2. Design

Published controller (`--ff`, alpha = 0.25, anticipation on, PER on), 1500 episodes, 500
held-out rollouts, seeds **0-19** in every arm.

| arm | W_PWR | status |
|---|---|---|
| A | 0.02 | to run |
| B | 0.05 | to run |
| C | 0.15 | **already on disk** - `runs/basin_v2` IS this arm; not re-run |
| D | 0.45 | to run |

**Seed reuse is correct here.** This is a treatment manipulation, so seeds 0-19 across arms means
common initial conditions. **The arms are NOT paired** - epsilon-greedy draws from the global RNG
once per step and episode length depends on the policy, so the streams diverge at the first
behavioural difference. Analyse unpaired.

**Not comparable across arms:** return, CVaR - the arms optimise different objectives.
**Comparable (physical):** pt_dbm, rlf_rate, steps, rb, ho_per_ep.
**Basin label:** pt_dbm < 27.0 dBm, not re-tuned.

## 3. The quantitative prediction, computed from basin_v2 only

For a fixed policy the return is linear in W_PWR:

```
return(W) = ret_ex_power - W * pnorm,  pnorm = steps * (pt - PT_MIN)/(PT_MAX - PT_MIN)
low  basin (n=6) :  89.920 - W *  3.607
high basin (n=14):  94.754 - W * 96.559
```

Equal at **W* = 0.0520**. Predicted ordering:

| W_PWR | basin the objective favours | margin |
|---|---|---|
| 0.02 | **HIGH** | +2.98 |
| 0.05 | dead heat | +0.19 to high |
| 0.15 | LOW | +9.11 |
| 0.45 | LOW | +36.99 |

*Limitation, stated in advance:* this predicts which basin has higher **objective value** under a
policy held fixed. It does not predict which basin **training will find**, and the whole point of
the paper is that they do not follow it reliably.

## 4. Predictions and adjudication

> **W1 (structure).** In any arm with >= 3 runs on each side, the empty interval is **>= 4 dB** and
> **zero** runs land strictly between 25 and 28 dBm. Vacuous for a unanimous arm.
>
> **W2 (propensity monotone).** n_low(0.02) <= n_low(0.05) <= n_low(0.15) = 6 <= n_low(0.45).
> *Recorded as a WEAK test: at n = 20 the standard error on a 0.30 proportion is +/-0.10, so
> differences of +/-2 runs mean nothing. Only the overall direction is tested.*
>
> **W3 (reliability consequence).** In any arm with >= 3 runs each side, low-basin mean RLF exceeds
> high-basin by **>= 5 pp**.
>
> **W4 (the decisive one).** At W_PWR = 0.02, where the objective clearly prefers high power,
> **at least one run still lands in the low-power basin.** If this holds, the bimodality is not
> explained by the reward preferring the low-power policy.

**KILL RULE, fixed now.** If arms A (0.02) and D (0.45) are each unanimous - 20/20 in opposite
basins - then W_PWR alone determines the outcome, bimodality exists only in a narrow band around
W* = 0.052, and **the paper's central claim must be reframed as a reward-indifference phenomenon
rather than a property of the environment.** That result is to be reported, not buried. No rescue
arm, no re-tuned threshold, no post-hoc reweighting.

**If W2 fails (non-monotone)** the mechanistic account in section 3 is wrong; stop and diagnose
before reporting anything.

## 5. Pre-flight completed before the sweep

1. **W_PWR reaches only the reward.** It appears in the environment at its definition and in the
   reward, and nowhere else - not in the observation. A clean single-variable manipulation.
2. **The override is exact.** `wpwr_override_proof.py` runs the same env, same reset seed, same
   fixed action sequence under two weights and asserts at every step that
   `r(w2) - r(w1) == -(w2-w1)*(pt-PT_MIN)/(PT_MAX-PT_MIN)`. Worst residual **1.9e-15**;
   trajectory identical; other weights unmoved; unset => 0.15 exactly.
3. **Three guards against a silent no-op.** The trainer aborts before training if the weight did
   not take; the active weights print into every job log; and `w_pwr` is written into every
   metrics file, which the driver re-reads after every job.
4. **Geometry cache warm**, so parallel workers cannot race on its writes.
5. **Reproduction control.** Seed 28 re-run at W_PWR = 0.15 through the new code path must
   reproduce the stored run exactly.
