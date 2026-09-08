# Pre-registration — the cross-algorithm arm (PPO)

**Written 8 September 2026, before any PPO run exists.** Standing project rule: a result is
reportable as a tested prediction only if the hypothesis, the threshold and the adjudication rule
were written down and dated before the runs they govern were trained. Nothing below may be revised
once the first PPO run has been trained; if it is revised, the arm becomes exploratory and says so.

---

## 1. The question this arm exists to answer

Every learned run in the paper comes from one algorithm family: a branched value-based agent with
dueling heads, quantile-regression critics, *n*-step returns and prioritised replay. §VIII states
that this is a limitation, and the reference policies added in §III cannot address it — they are not
learned. The open question is therefore exactly one thing:

> **Is the bimodal convergence structure a property of this environment and objective, or a property
> of the value-based algorithm family used to optimise them?**

A policy-gradient method is the natural contrast. PPO optimises a stochastic policy by clipped
surrogate ascent with no replay buffer, no target network, no bootstrapped value backup over a
discrete action-value table and no risk measure — it shares almost none of the machinery that §IV-D
and §IV-G eliminated as explanations, and none of the machinery that §IV-H implicates.

## 2. What is held identical

Everything except the learner:

- the environment, `paper2_env.MultiRSUBlockageEnv`, unmodified;
- the reward, unmodified, at the published weights (1.0, 20.0, 0.15, 0.05, 0.05);
- the action space, `MultiDiscrete([2, 3, 3])` — the published `inc3` power actuator;
- the training/held-out scene split (30 / 15);
- the evaluation: **the same `evaluate()` and `summarize()` functions the paper's own runs used**,
  500 held-out episodes, common random numbers at `EVAL_SEED_BASE = 900,000,000`, greedy actions;
- the training budget: **1500 episodes per run**, as in every learned arm of this paper;
- the seeds: 0–19, as in every other arm.

The PPO agent exposes the same `act(obs, hidden, eps)` interface as the published controller, so the
evaluation path is byte-for-byte the one used elsewhere and cannot drift.

## 3. Sample size

Twenty seeds, 0–19, matching every other arm. **The one quantity measured before this number is
fixed is wall-clock time per run**, from a single smoke run; no outcome — no transmit power, no
reliability, no return — is inspected before the twenty runs are launched. If the measured wall
clock makes twenty infeasible on the available hardware, the number is reduced *before* launch, the
reduced number is recorded here with the measurement that forced it, and the loss of power is
reported in the paper.

## 4. Predictions

**X1 — decisive, the structure.** Sorting the twenty PPO run means of transmit power over the
held-out evaluation leaves an empty interval of **at least 4 dB**, and **no run** finishes with a
mean between 25 and 28 dBm.

**X2 — the consequence, conditional on X1.** If X1 holds, sustained radio-link failure in the
low-power regime exceeds that in the high-power regime by **at least 5 percentage points**.

**X3 — the classifier.** The regime threshold is the **27 dBm of §IV-A, not re-tuned**, and the
classification is invariant to any cut between 24 and 29 dBm.

The 4 dB in X1 is the bound already used for W1 in the §IV-E pre-registration, chosen there and
reused here rather than set for this arm.

## 5. The competence gate, registered in advance

A learner that never learned says nothing about where learning converges. The arm is **informative
only if PPO is a working controller**, judged on a bar fixed now:

> **Gate G.** The median over seeds of held-out sustained radio-link failure must be **below
> 12.40 %**, the rate achieved by the uniform-random policy measured in §III.

This is deliberately a low bar. It excludes only the case "PPO did not learn"; it does not require
PPO to match the published controller, and passing it is not a claim that PPO is competitive.

**If Gate G fails at 1500 episodes**, one fallback is permitted and is registered here rather than
invented later: a single re-run of the same twenty seeds at **6000 episodes** (four times the
budget, reflecting that PPO is on-policy and less sample-efficient than replay-based learning). If
Gate G fails again, the arm is declared **uninformative** and reported as such — no further budget
increases, no architecture search, no hyper-parameter tuning against the outcome.

## 6. What each outcome means, decided now

| Outcome | Reading | Consequence for the paper |
|---|---|---|
| Gate G passes, **X1 holds** | The structure survives a change of algorithm family. It is a property of the environment and objective, not of value-based learning. | §VIII's algorithm-family limitation is substantially narrowed. The paper's claim generalises one step and says so, still within one environment. |
| Gate G passes, **X1 fails** — PPO is unimodal | The structure **is** specific to the value-based family. | **This is a larger and different result than the present paper.** It would make the finding a statement about algorithm choice as a reliability-relevant design decision, and the paper must be reframed around it rather than reporting it as an aside. It also makes §IV-H's action-representation account more interesting, not less: the actuator is addressed identically here, so an algorithm-family difference would mean the actuator is necessary but not sufficient. |
| Gate G passes, X1 partially holds (a gap, but narrower than 4 dB, or one run inside) | Ambiguous. | Reported as a failed prediction with the observed values, exactly as the six failures in §IV-I are. No re-threshold. |
| **Gate G fails twice** | PPO did not learn this task at either budget. | Arm reported as **uninformative**. The §VIII limitation stands unchanged. This outcome is declared here in advance so that it cannot later be presented as evidence of unimodality. |

## 7. Analysis, fixed now

- The experimental unit is the training run.
- Regime assignment: mean transmit power over the 500 held-out episodes, threshold 27 dBm.
- Empty interval: highest low-regime run mean to lowest high-regime run mean, from unrounded values.
- X2 is tested only if both regimes are non-empty; if one regime is empty, X2 is **unevaluable**
  and is reported as unevaluable, not as held or failed.
- No statistic other than those named above is computed for adjudication. Anything else that emerges
  is exploratory and is labelled exploratory.

## 8. Declared possible null

**It is a legitimate and publishable outcome of this arm that PPO shows no bimodality**, and it is
the outcome that would most change the paper. It is recorded here in advance precisely so that it
cannot be quietly reframed if it occurs.

---

*Recorded by G. Dayasagar, 8 September 2026, before the first PPO run.*
