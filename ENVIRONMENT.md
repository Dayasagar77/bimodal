# Environment and seeding

This file backs the paper's claim that a run is a deterministic function of
(seed, configuration). It records the exact interpreter and package versions every run in this
repository was produced under, and names the call sites at which the run seed enters the pipeline.

## Interpreter and packages

    python      3.11.9
    platform    Windows-10-10.0.26200-SP0
    cpu         AMD64
    torch       2.5.1+cu121
    numpy       2.4.4
    pandas      3.0.3
    gymnasium   1.2.3
    scipy       1.17.1

Captured at sweep time; the same record ships with each sweep as `ENVIRONMENT.txt`.

**On `+cu121`.** The installed PyTorch wheel is the CUDA build, but no run in this paper used a GPU.
The controller is constructed on the CPU and nothing ever moves a tensor off it:

    paper2_controller.py:176   def __init__(self, obs_dim, alpha=CVAR_ALPHA, lr=5e-4, device='cpu', use_lstm=True)

`grep -n "cuda\|is_available" paper2_controller.py paper2_env.py` returns nothing. There is no
device-selection branch to take, so no GPU or cuDNN kernel is involved and the usual sources of
non-deterministic reduction order do not arise.

**Threads.** PyTorch is held to a single thread, which is the default rather than an option that has
to be remembered:

    paper2_controller.py:496   ap.add_argument('--threads', type=int, default=1,
                                               help='torch CPU threads (1 = deployment condition)')
    paper2_controller.py:518   torch.set_num_threads(max(1, a.threads))

## Where the seed enters

The run seed fixes the whole pipeline, not merely the network initialisation. Every call site:

| what | call site |
|---|---|
| network initialisation, exploration draws | `paper2_controller.py:560` — `torch.manual_seed(a.seed); np.random.seed(a.seed)` |
| training environment (scene draw, shadow fading, small-scale fading) | `paper2_controller.py:551` — `MultiRSUBlockageEnv(pool, seed=a.seed, …, split='train')` |
| held-out environment | `paper2_controller.py:552` — `MultiRSUBlockageEnv(pool, seed=a.seed + 10_000, …, split='test')` |
| prioritised replay sampling | `paper2_controller.py:576` — `rng=np.random.default_rng(a.seed + 777)` |
| the environment's own generator | `paper2_env.py:177` — `self.rng = np.random.default_rng(seed)` |
| per-episode determinism | `paper2_env.py:278` — `reset(seed=k)` re-seeds the generator, so `k` fixes both the scene and the entire fading realisation |

Two seeds are deliberately **not** derived from the run seed, so that they are common across every
run and every configuration:

| what | call site |
|---|---|
| train/held-out scene split | `paper2_env.py:131` — `SPLIT_SEED = 20260725`, used at `:142`; "fixed forever; the split must not move with the training seed" |
| the 500 evaluation episodes | `paper2_controller.py:426` — `env.reset(seed=EVAL_SEED_BASE + i)`; episode *i* always uses the same reset seed, so every configuration and every training seed is scored on exactly the same 500 scenarios with the same fading realisations |

## What was checked, and how

Re-running seeds 0–19 under an unchanged configuration, after a source change that added per-episode
instrumentation, reproduced all twenty runs to four decimal places on sustained RLF, mean return,
CVaR25 return and mean transmit power; a five-episode controlled comparison against the
pre-instrumentation source produced byte-identical return series. That is the evidence for the
determinism claim. The table above says where the determinism comes from; the re-runs are what
establish that nothing outside it leaks in.

## Source fingerprints

The two files the table refers to, as of the runs in this repository:

    md5  bd14f06763852611056505e696ce24f3  paper2_controller.py
    md5  f916da4619fe21db56056d66142bf49f  paper2_env.py

Both are public in the environment and controller repository:
https://github.com/Dayasagar77/Anticipate
