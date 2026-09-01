# Pre-registrations

Reproduced **unedited** from the project record, with their original dates. Each was written
before the runs it describes existed. They are working documents, not polished prose: they name
what would count as failure, they record honest priors, and where a prediction was weak they say
so in advance. Editing them after the fact would defeat the point of having them.

| file | written | governs |
|---|---|---|
| `basin_sweep_20260828.md` | 28 Aug 2026 | the twenty fresh seeds of SIV-C (P1 structure, P2 consequence, P3 proportion) and the single-cell detection test of SVI-B (P4) |
| `wpwr_sweep_20260830.md` | 30 Aug 2026 | the four-arm reward-weight sweep of SIV-E (W1-W4 and the kill rule) |
| `pwrform_sweep_20260830.md` | 30 Aug 2026 | the power-penalty **form** sweep of SIV-G (Q1-Q4 and the rails-unmoved kill rule) |
| `pwract_sweep_20260830.md` | 30 Aug 2026 | the power-**actuator** sweep of SIV-H (A1-A4), which identified the cause |

All four were written before the corresponding `runs/` directories existed.

**Eight predictions failed and all eight are reported as failures in the paper:** P4 (SVI-B),
W2 (SIV-E), all four of Q1-Q4 (SIV-G), and A3 and A4 (SIV-H).

Two of those failures are ours rather than the manipulation's, and are worth reading as such.

**SIV-G, Q4.** The rail bound was set from one arm's observed range with no allowance for
sampling variation. Both new arms breached it - by **0.022 dB and 0.126 dB**, against an empty
interval of roughly 5.9 dB. Those are not rails moving; they are bounds set too tightly. The rule
was not waived to rescue the arms and the bound was not pretended to hold: **both form arms are
reported as exploratory rather than confirmatory**, and the breach and its magnitude are stated in
the paper.

**SIV-H, A3.** The consequence test required interior reliability to fall between the low- and
high-basin rates *of the same arm*. In `direct8`, nine of the eleven "low-basin" runs **are** the
interior policies, because the 27 dBm classifier separates rails only in a bimodal arm. The test
is circular exactly where it matters, and it is reported as failed rather than quietly replaced
with the comparison that does work (against the published rails), which is reported separately and
labelled exploratory.

The point of writing a kill rule down in advance is that it still binds when it turns out to be
inconvenient.
