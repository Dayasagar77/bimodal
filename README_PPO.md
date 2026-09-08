# The cross-algorithm arm (PPO), 8 September 2026

    prereg/PREREGISTRATION_PPO_20260908.md   written and dated before the first run
    prereg/RESULT_PPO_20260908.md            the outcome, failures included
    analysis/ppo_arm.py                      trains and evaluates; reuses the paper's own
                                             evaluate()/summarize() so the held-out protocol
                                             is byte-for-byte the one used by every QR-DQN run
    analysis/adjudicate_ppo.py               applies the registered rules; written before any
                                             seed finished
    analysis/crosscheck_ppo.py               re-derives every PPO number in SIV-J, non-zero on
                                             a mismatch
    runs/ppo/                                20 seeds: per-episode held-out results, per-run
                                             metrics, training return series

Reproduce one seed:

    python analysis/ppo_arm.py --pool dair_geometry_pool.csv --episodes 1500 --seed 0 \
        --eval_n 500 --out runs/ppo
