# Result — the cross-algorithm arm (PPO)

**Run 8 September 2026 against `PREREGISTRATION_PPO_20260908.md`, which was written and dated
before the first PPO run existed.** Twenty seeds, 0–19, 1500 episodes each, identical
environment, reward, action space, scene split and evaluation path (the paper's own
`evaluate()` and `summarize()`, 500 held-out episodes, common random numbers at
EVAL_SEED_BASE = 900,000,000, greedy actions). Only the learner differs.

---

## The registered tests

| | prediction | outcome |
|---|---|---|
| **Gate G** | median sustained RLF below the random policy's 12.40 % | **PASS**, narrowly — 12.00 % |
| **X1** | empty interval ≥ 4 dB and no run inside 25–28 dBm | **HELD** — 4.741 dB, 0 runs inside |
| **X2** | low-regime RLF exceeds high-regime by ≥ 5 pp | **FAILED** — +3.48 pp (13.18 % vs 9.70 %) |
| **X3** | classification invariant to any cut in 24–29 dBm | **FAILED** — two runs at 28.09 and 28.51 dBm break it |

One of three registered predictions held. Both failures are reported here and go into §IV-I with
the other six.

## What was found

**The bimodal transmit-power structure reproduces under a policy-gradient learner.** Twelve of the
twenty runs converge to 23.25–23.35 dBm and eight to 28.09–30.00 dBm, with an empty interval of
**4.741 dB** and not one run inside the 25–28 dBm band fixed in the §IV-C pre-registration. PPO
shares none of the machinery the paper's controller uses — no replay buffer, no target network, no
quantile critic, no risk measure — and it is a stochastic policy optimised by clipped surrogate
ascent rather than a deterministic greedy policy over action values. The structure appears anyway.

**But it is a looser structure than the published controller's, in three measurable ways.**

1. *The regimes are not two points.* The published controller's high rail spans 29.13–30.00 dBm;
   PPO's spans 28.09–30.00, and the two runs at 28.09 and 28.51 are what break threshold
   invariance (X3). The distribution is bimodal but not degenerate.
2. *The reliability consequence is weaker.* +3.48 pp against the ≥ 5 pp registered, and the
   correlation between mean transmit power and sustained RLF is −0.440 against −0.920 for the
   published controller. Rank separation is **not** complete: the high regime contains both the two
   best runs in the whole arm (0.80 % and 1.80 %) and five runs at 10.60 %.
3. *PPO is a much weaker controller.* Median sustained RLF 12.00 %, against 1.03 % for the
   published controller's high regime and 10.40 % for the best conventional rule of §III. It clears
   the competence gate by 0.40 pp. **This arm shows the structure appears in another algorithm
   family; it does not show it appears in a competitive controller of that family.** A stronger PPO
   — more budget, tuned hyper-parameters, a recurrent policy — might behave differently, and this
   arm does not speak to that.

## Exploratory, and labelled exploratory

The handover branch splits too, and not in step with the power branch: six runs settle at more than
ten handovers per episode and thirteen at under one, with one in between. Crossed with the power regime that is four
behaviours, not two — which is the same shape as the two independent lotteries reported in §IV-F for
power and bandwidth, now on a different pair of actuators and a different algorithm. This was not
registered, was not predicted, and is reported as an observation only.

## What this changes in the paper

§VIII's "one environment, one algorithm family" limitation is narrowed on the algorithm axis and
kept on every other. The claim the paper can now make is that the structure is not an artefact of
value-based learning, established on one policy-gradient learner at one budget in one environment.
The claim it still cannot make is that the structure is universal, or that it survives in a
competitive controller of a different family.

## Data

`runs/ppo/` — per-episode held-out results, per-run metrics and the training return series for each
of the twenty seeds. `analysis/ppo_arm.py` trains and evaluates; `analysis/adjudicate_ppo.py`
applies the registered rules and prints the verdict. The adjudicator was written before any seed
finished.
