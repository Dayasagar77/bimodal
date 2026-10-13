# Pre-registrations

Reproduced **unedited** from the project record, with their original dates. Each was written
before the runs it describes existed. They are working documents, not polished prose: they name
what would count as failure, they record honest priors, and where a prediction was weak they say
so in advance. Editing them after the fact would defeat the point of having them.

| file | written | governs |
|---|---|---|
| `basin_sweep_20260828.md` | 28 Aug 2026 | the twenty fresh seeds of sweep B (P1 structure, P2 consequence, P3 proportion) and the single-cell early-detection test (P4) |
| `wpwr_sweep_20260830.md` | 30 Aug 2026 | the four-arm reward-weight sweep (W1-W4 and the kill rule) |
| `pwrform_sweep_20260830.md` | 30 Aug 2026 | the power-penalty **form** sweep, the paper's penalty-shape arms (Q1-Q4 and the rails-unmoved kill rule) |
| `pwract_sweep_20260830.md` | 30 Aug 2026 | the power-**actuator** sweep (A1-A4) |
| `PREREGISTRATION_PPO_20260908.md` | 8 Sep 2026 | the PPO arm (X1-X3 and the competence gate); its outcome, failures included, is `RESULT_PPO_20260908.md` |

All five were written before the corresponding `runs/` directories existed.

**Eight predictions failed and all eight are reported as failures in the paper:** P4, W2,
Q1, Q2 (on the empty interval; its count held), Q4, A3, and the PPO arm's X2 and X3. Q3 held
where it applied, and A4 held for `inc7` and could not be evaluated for `direct8`.

Two of those failures are ours rather than the manipulation's, and are worth reading as such.

**Penalty shape, Q4.** The rail bound was set from one arm's observed range with no allowance for
sampling variation. Both new arms breached it - by **0.022 dB and 0.126 dB**, against an empty
interval of roughly 5.9 dB. Those are not rails moving; they are bounds set too tightly. The rule
was not waived to rescue the arms and the bound was not pretended to hold: **both form arms are
reported as exploratory rather than confirmatory**, and the breach and its magnitude are stated in
the paper.

**Actuator, A3.** The consequence test required interior reliability to fall between the low- and
high-basin rates *of the same arm*. In `direct8`, nine of the eleven "low-basin" runs **are** the
interior policies, because the 27 dBm classifier separates rails only in a bimodal arm. The test
is circular exactly where it matters, and it is reported as failed rather than quietly replaced
with the comparison that does work (against the baseline arm's regimes), which the paper reports
separately.

The point of writing a kill rule down in advance is that it still binds when it turns out to be
inconvenient.
