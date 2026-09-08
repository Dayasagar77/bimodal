# Additions for the 8 September 2026 revision

Copy these into the repository root, keeping the directory names, then commit and push.
The manuscript's Data-and-code-availability section names each of them.

    runs/baselines/            13 reference policies on the same 500 held-out episodes
                               (CRN base 900,000,000), one per-episode file each plus
                               baselines_metrics.csv
    analysis/paper2_baselines.py    the evaluator; adds the fixed-low-power mode used in Table III
    analysis/crosscheck_baselines.py  re-derives every cell of Table III and exits non-zero on a mismatch
    analysis/make_figures_v2.py     regenerates Figs 1, 2, 4, 5 and the new Fig 6
    figures/                   the revised figure PDFs

After pushing, replace `<COMMIT>` in the manuscript's repository citation with the new short hash.
