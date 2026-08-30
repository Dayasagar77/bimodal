# PRE-REGISTRATION — basin_v2 sweep

**Written 28 August 2026, BEFORE the sweep is launched.** This discharges the test the basin analysis
has been owed since 23 August: *"fix the basin threshold and the predicted propensities in advance,
train a fresh set of seeds, and see whether the proportions land where this analysis says they will."*

**Run:** 20 seeds x 1500 episodes, reported config (FF + anticipation + CVaR 0.25 + PER), held-out
evaluation, `models/basin_v2/model_s{0..19}`. Uses the patched `paper2_controller.py`, verified a pure
observer: byte-identical returns against the pre-patch backup under identical arguments.

---

## Predictions, strongest first

**P1 - STRUCTURE (the sharp one).** Sorting the 20 mean transmit powers leaves an **empty interval of
at least 5 dB**, and **any threshold between 24 and 29 dBm classifies all 20 runs identically.**
Falsified if any run finishes between 25 and 28 dBm, or if the threshold choice changes any label.

**P2 - CONSEQUENCE.** Low-basin sustained RLF exceeds high-basin RLF **by at least 5 pp**. The 23 Aug
figures were 8.90% against 0.96%.

**P3 - PROPORTION (the weakest, and stated as weak).** Roughly **6 of 20** runs land in the low-power
basin. With n = 20 the standard error on a 0.30 proportion is +/-0.10, so anything from about **2 to 10**
is consistent and this prediction can barely fail. It is recorded for completeness, not as a test.

**P4 - EARLY DETECTION FROM POWER.** Re-running `basin_early_detect.py` against the new
`pt_mean` column gives within-config AUC **>= 0.80 at a window ending <= episode 500**, family-wise
p < 0.05. Return reached only 0.714 at that horizon. If power does no better, the basin is not
early-predictable from either observable and the steering line closes.

*Note on P4: a single-config sweep means within-config equals pooled, so the confound control that
mattered across eight configs does not arise here. State that when reporting rather than implying a
control was applied.*

## What each outcome means

| | |
|---|---|
| P1 + P2 hold | The basin structure is **confirmed on fresh data**, no longer post-hoc. That alone upgrades a 23 Aug exploratory observation into a result. |
| P1 fails | The bimodality was an artefact of the original sweep. Serious - it would also qualify the per-seed subsection drafted for the companion paper. |
| P4 holds | A detector exists on the defining quantity. **Steering** becomes the contribution; the detector is a baseline, not the result. Prior-art gate runs before any design. |
| P4 fails | Neither return nor power predicts the basin early. The steering line closes and the honest output is a negative result: *a 9x reliability lever that is real, consequential, and not predictable from the observables available during training.* |

## Analysis fixed in advance

- Threshold **27 dBm**. Not tuned.
- `basin_early_detect.py --dir models/basin_v2`, windows 100/200/300/500/750, the same six features,
  max-statistic permutation across all cells, 20,000 iterations.
- No feature, window, or threshold may be added after seeing the result. If something interesting
  appears outside this list it is exploratory and must be labelled so.

## Honest prior

The last four candidates died on measurement, and this is the fifth measurement in a chain: 9x lever ->
not detectable from return -> maybe detectable from power -> maybe steerable. **P4 is roughly a coin
flip in my estimation, and even if it holds, steering remains unproven and ungated.** The value of
this run does not depend on P4: P1 and P2 alone convert a post-hoc observation into a confirmed one,
which is worth the three hours regardless.
