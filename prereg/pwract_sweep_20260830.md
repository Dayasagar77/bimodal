# PRE-REGISTRATION — is the empty interval a property of the ACTUATOR?

**Written 30 August 2026, BEFORE either arm was trained.** No run of `direct8` or `inc7`
existed at the time of writing. The only run executed beforehand is an `inc3` reproduction
control on seed 0, whose sole purpose is to prove the patch is inert when unset.

---

## 1. Why this exists

Three explanations of the empty 25-28 dBm interval have now been tested:

| account | status |
|---|---|
| Risk-averse local-optimum barrier (Greenberg et al., NeurIPS 2022) | **refuted** - removing CVaR *doubles* the split, 6/20 -> 13/20 (SIV-D) |
| Linear scalarisation's non-convexity (Vamplew et al., 2008) | **not supported** - pricing the interior cheaper gave 0/20 interior runs (SIV-G) |
| Reward *weight* | **not supported** - 22-fold range, 0/80 interior runs (SIV-E) |

What remains is the account recorded in the SIV-G pre-registration before those runs existed:
**the actuator.** The published power branch is a three-action increment/decrement control with
hard limits, and the episode resets at the ceiling. Reaching the floor requires **seven
consecutive decrements**. A deterministic greedy policy over such a branch may saturate at a
limit not because the interior is dominated but because it is never durably represented.

This pre-registration tests that by changing the **addressing** while holding the reachable
power set fixed.

## 2. Prior-art gate

Run before designing, on 30 Aug 2026.

- **Seyde et al., "Is Bang-Bang Control All You Need?" (NeurIPS 2021)** is the closest work and
  the most important one. Verified against the proceedings PDF: their bang-bang is about
  **actions taking extreme values**, they use **absolute** action addressing throughout, they do
  **not** study incremental/delta parameterisations, and they report **no seed multimodality**.
  They attribute emergence partly to benchmarks *without action penalties* - we have one, and
  SIV-E shows a 22-fold variation of it does not dissolve the structure.
  **This is evidence against our hypothesis** and is weighted as such in S5 below: their agents
  address actions directly and still go extremal.
- **"Demystifying Action Space Design for Robotic Manipulation Policies" (ICML 2026)** compares
  delta against absolute action spaces and finds delta *outperforms* absolute, explained via
  noise amplification. It reports no seed-variance, multimodality, or saturation analysis.
- No work found that links an incremental actuator to **state saturation producing seed-level
  bimodal convergence**, and none in the wireless/RRM DRL literature comparing power-step
  against direct-power-level action spaces for this effect.

**Gate verdict: the specific claim is open.** Seyde et al. becomes a citation that sharpens the
framing and supplies the strongest reason to doubt our own hypothesis.

## 3. The manipulation, and the property that makes it clean

| mode | actions | semantics | worst-case steps to cross the range |
|---|---|---|---|
| `inc3` | 3 | delta {-1, 0, +1} dB, then clip - **the published actuator** | **7** |
| `inc7` | 7 | delta {-3 ... +3} dB, then clip | **3** |
| `direct8` | 8 | absolute level, *a* -> 23 + *a* dBm | **1** |

**All three reach exactly the same power set, {23, ..., 30} dBm on a 1 dB grid.** Proved before
launch by `pwract_proof.py`, which computes the reachable closure from the reset state under
each mode and confirms the three sets are identical; it also confirms `inc3` reproduces the
published expression `clip(pt + (a-1)*PT_STEP, PT_MIN, PT_MAX)` **exactly**, worst difference
`0.0` over every (power, action) pair, so an unset `P2_PWR_ACT` is inert.

`inc7` is the **control**. It shares `direct8`'s fast traversal and near-equal action count while
remaining incremental, so a `direct8`-vs-`inc7` difference isolates **addressing** from branch
width and traversal speed.

**Stated confound, not hidden:** direct addressing also changes what epsilon-greedy explores - a
random policy over 8 absolute levels samples the interior constantly, where a +/-1 random walk
does not. Reachability and exploration coverage are **entangled by construction**. This
experiment tests the actuator as a whole; it cannot separate those two, and no claim will be made
that it does.

**Run:** published controller (`--ff`, alpha = 0.25, anticipation on, PER on), W_pwr = 0.15,
linear form, 1500 episodes, 500 held-out rollouts, seeds **0-19** per arm. `inc3` is
`runs/basin_v2`, on disk and not re-run. Seeds shared across arms for common initial conditions;
arms analysed unpaired.

## 4. Predictions

The SIV-G result changed how the outcome must be measured. A run's *mean* transmit power landing
inside the interval is **not** sufficient - sqrt seed 16 did that while alternating between rails
across episodes. Every prediction below is therefore stated at the **policy** level.

> **Definition - committed interior policy.** A run whose mean transmit power lies strictly in
> 25-28 dBm **and** more than **50%** of whose held-out episodes have their own mean inside
> 25-28 dBm. Across all 120 runs analysed to date the maximum is **27%**, so this bar has never
> been met, by any run, under any weight or reward form.

> **A1 - DECISIVE.** Under `direct8`, **at least one of the twenty runs is a committed interior
> policy.** If the interval is empty because the incremental actuator never durably represents
> interior policies, direct addressing should produce at least one.
>
> **A2 - ATTRIBUTION.** `direct8` yields **strictly more** committed interior policies than
> `inc7`. If the two match, the operative variable is traversal speed or branch width, not
> addressing, and the paper must say so.
>
> **A3 - CONSEQUENCE.** Any committed interior policy has sustained RLF strictly between the
> pooled low-basin and high-basin rates of its own arm. RLF is monotone in power everywhere we
> have measured it (corr -0.638 to -0.920 across six configurations); an interior policy with
> rail-like reliability would mean that relation is not what we think it is.
>
> **A4 - RAILS.** Runs that still classify into a basin keep their locations: committed
> low-basin runs finish at **<= 23.6 dBm**, committed high-basin runs at **>= 28.9 dBm**.

**On A4's bounds.** In SIV-G this kind of bound was set from one arm's observed range with no
allowance for sampling variation, and it was breached by 0.022 dB and 0.126 dB - a specification
error that cost two arms their confirmatory status. The bounds here are set from the **full
120-run record**: observed low maxima span 23.24-23.53 dBm and high minima 28.98-29.71 dBm. The
bounds above clear the worst observed value by roughly 0.07 dB on each side while still catching
any genuine rail shift, which would be on the order of 1 dB. If they are breached by less than
0.25 dB the arm is **not** voided; the breach is reported and the analysis proceeds. Anything
larger voids the arm as before.

## 5. Honest prior

**A1 at about 40%, and the single strongest reason to doubt it is Seyde et al.:** their agents
address actions directly and still converge to extremes. Direct addressing demonstrably does not,
on its own, prevent extremal policies.

There is also a fourth explanation this experiment would leave standing, and it deserves naming
now rather than being constructed afterwards if A1 fails: **the interior may simply be
dominated.** SIV-E's fitted returns give a low basin of 89.920 - W*3.607 and a high basin of
94.754 - W*96.559, and at no weight tested does an interior policy win. If a sharp sub-THz SINR
cliff makes intermediate power a genuinely worse operating point, then bimodal convergence is a
*correct* response to a bimodal optimum landscape, and the paper's contribution changes shape:
from "a pathology of training" to "a real property of the objective landscape that seed-averaged
reporting conceals." That is still a publishable and honest result, and it is the one expected if
both new arms come back empty.

**No actuator beyond `inc7` and `direct8` will be added after seeing these results.** If a third
is ever run it is exploratory and must be labelled so. No re-tuned threshold, no post-hoc
redefinition of "committed."

## 6. Verification chain, all executed before launch

1. `P2_PWR_ACT` read at **module scope** - `paper2ma_env.py` copies these names at import, so a
   post-import assignment would silently fail to propagate.
2. `inc3` reproduces the published power expression exactly (worst difference `0.0`).
3. The three reachable power sets are identical, computed by closure from the reset state.
4. **G1a** - the trainer aborts before episode 1 if the requested actuator did not take effect.
5. **G1b** - the trainer aborts if the network's branch widths disagree with the environment's
   action space, on either the training or the held-out environment.
6. **G2** - every result file records `pwr_act` and `pwr_n_act`; the sweep driver re-reads it
   after each run and counts mismatches.
7. **Reproduction control** - one full `inc3` run at seed 0 through the new code path, diffed
   against the stored `basin_v2` seed 0 on every recorded column. Result: bit-identical on the
   500-row held-out evaluation and the 1500-row return trace, worst difference 0.0.

## 7. Files

- `analysis/patch_pwract_20260830.py` - the patch (idempotent, backs up both files)
- `analysis/pwract_proof.py` - reachable-set, inertness and traversal-cost proof
- `analysis/pwract_sweep.py` - the driver
- `analysis/pwract_adjudicate.py` - scores A1-A4 against the runs
- outputs: `runs/pwract_direct8/`, `runs/pwract_inc7/`
