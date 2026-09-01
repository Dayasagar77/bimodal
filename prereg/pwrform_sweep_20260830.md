# PRE-REGISTRATION — does the *form* of the power penalty create the empty interval?

**Written 30 August 2026, BEFORE either arm was trained.** No run of `quad` or `sqrt`
existed at the time of writing.

---

## 1. Why this exists

Two published theories predict two attractors in this setting. One has already been tested
and fails:

- **Risk-averse local-optimum barrier** (Greenberg et al., NeurIPS 2022). Proves that
  because risk measures concentrate on worst-case returns, risk-averse RL ignores
  high-return strategies and this "inevitably leads to a local-optimum barrier."
  **Ruled out by the existing ladder:** removing CVaR entirely and optimising the
  expectation *doubles* the low-power basin, 6/20 -> 13/20. Deleting the named mechanism
  makes the effect stronger.
- **Linear scalarisation cannot reach non-convex Pareto regions** (Vamplew et al., 2008).
  Two well-separated policies can tie in scalarised value with nothing achievable between
  them. The reward here is exactly a linear scalarisation of availability against power.
  **Untested.** SIV-E varied the penalty's *weight* over a twenty-two-fold range; its
  *form* has never been varied.

This pre-registration tests the second.

## 2. The manipulation, and the property that makes it clean

With *u* = (P_t - P_min)/(P_max - P_min), the penalty becomes *W*.f(u) for

| form | f(u) | effect on interior |
|---|---|---|
| `linear` | *u* | the published reward |
| `quad` | *u*^2 | interior **cheaper** than linear |
| `sqrt` | sqrt(*u*) | interior **dearer** than linear |

**Every form satisfies f(0) = 0 and f(1) = 1.** The two rail policies are therefore priced
**identically** under all three, and only intermediate-power policies are repriced. This is
the whole point of the design: any change in outcome cannot be attributed to the rails
being made cheaper or dearer, only to the interior becoming reachable or not.

Verified before launch by `pwrform_proof.py`: floor and ceiling penalties bit-identical
across the three forms (0.000000 and 0.150000), `quad` strictly cheaper and `sqrt` strictly
dearer at all seven interior points, and unset => `linear` exactly.

**Run:** published controller (`--ff`, alpha = 0.25, anticipation on, PER on), *W*_pwr = 0.15,
1500 episodes, 500 held-out rollouts, seeds **0-19** per arm. Two new arms, `quad` and
`sqrt`; `linear` is `runs/basin_v2`, already on disk and not re-run. Seeds shared across
arms give common initial conditions; the arms are **not** paired and are analysed unpaired,
for the reasons given in the weight-sweep pre-registration.

## 3. Predictions

> **Q1 — DECISIVE.** Under `quad`, **at least one of the twenty runs finishes strictly
> between 25 and 28 dBm.** If the empty interval exists because linear scalarisation leaves
> interior policies dominated, then pricing the interior cheaper while leaving both rails
> untouched should put runs inside it.
>
> **Q2 — DIRECTION.** Under `sqrt`, the interval stays empty **and** the low-basin count is
> not below `linear`'s 6/20. This is the mirror control: the effect must run in the
> direction the mechanism predicts, not merely appear.
>
> **Q3 — CONSEQUENCE.** In any arm with >= 3 runs on each side of 27 dBm, low-basin
> sustained RLF exceeds high-basin by **>= 5 pp** — the same bar as P2 and W3.
>
> **Q4 — RAILS UNMOVED.** Because the rails are priced identically, rail *locations* are
> unchanged: low-basin runs finish within 23.2-23.4 dBm and high-basin within 29.0-30.0.
> A rail that moves means the manipulation did more than intended and the arm is void.

## 4. Adjudication, fixed now

| outcome | reading |
|---|---|
| **Q1 holds, Q2 holds** | The linear form is implicated. The paper gains a *mechanism*: the empty interval is a property of the scalarisation geometry, not of the algorithm or the environment. Major upgrade. |
| **Q1 fails** (interval empty under both forms) | Linear scalarisation is ruled out **alongside** the CVaR barrier. The paper will then have eliminated the two best published accounts of its own phenomenon. That is a real contribution and will be reported as one; the mechanism stays open. |
| **Q1 holds, Q2 fails** (`sqrt` also populates the interval) | The effect is not directional and something other than scalarisation geometry is moving. Stop and diagnose before reporting anything. |
| **Q4 fails** | The manipulation moved the rails; the arm is void regardless of Q1. |

**No form beyond `quad` and `sqrt` will be added after seeing these results.** If a third
is ever run it is exploratory and must be labelled so. No re-tuned threshold, no post-hoc
reweighting.

## 5. Honest prior

**I put Q1 at roughly one in three.** There is a third candidate mechanism that neither
Greenberg nor Vamplew covers and that this experiment cannot touch: the power branch is a
three-action increment/decrement actuator with hard limits, and a deterministic greedy
policy over such an actuator tends to saturate at a rail regardless of how the interior is
priced. If the empty interval is bang-bang saturation, both new arms will come back empty
and Q1 fails — not because scalarisation is innocent, but because the actuator never
offered an interior policy to begin with.

That alternative is worth stating in advance because it is the most likely explanation of a
null result here, and because it suggests the next experiment (a finer or direct-set power
actuator) rather than leaving a null looking like a dead end.

## 6. Files

- `analysis/patch_pwrform_20260830.py` — the patch (idempotent, backs up both files)
- `analysis/pwrform_proof.py` — the rails-identical / interior-repriced proof
- `analysis/pwrform_sweep.py` — the driver
- outputs: `runs/pwrform_quad/`, `runs/pwrform_sqrt/`
