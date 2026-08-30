#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_wpwr_20260830.py -- add a W_PWR override to the Paper-2 env and controller.

WHY AN ENV VAR AND NOT JUST `PE.W_PWR = x`. paper2_env.py reads W_PWR as a module global
inside step(), so a post-import assignment WOULD work there. paper2ma_env.py line 51 COPIES
the value at import time, so a post-import assignment would silently do nothing for the MA
env. Reading the override at module scope in paper2_env.py is immune to import order in both
files, so the same mechanism cannot rot if this is ever reused.

THREE GUARDS, because a silent no-op here wastes hours:
  G1  paper2_controller.py asserts PE.W_PWR equals what was asked, immediately after import,
      before a single episode runs.
  G2  the active weight is printed in the config banner, so it lands in every job log.
  G3  the active weight is written into _metrics.csv as a column, so every result file
      self-documents the reward it was trained under. An arm that silently failed to apply
      the override is then visible in the data, not merely absent from it.

Default behaviour is unchanged: with P2_W_PWR unset, W_PWR is 0.15 exactly as before, so
every existing script and every stored result stays valid.

Idempotent: re-running detects the markers and does nothing.
"""
import os, re, shutil, sys

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Paper2', 'code')
CODE = os.path.abspath(CODE)
STAMP = '20260830'


def patch(path, anchor, insert, marker):
    src = open(path, encoding='utf-8').read()
    if marker in src:
        print(f'  SKIP  {os.path.basename(path)}: already patched ({marker})')
        return False
    n = src.count(anchor)
    if n != 1:
        sys.exit(f'  ABORT {os.path.basename(path)}: anchor matched {n} times, expected 1\n'
                 f'        anchor: {anchor[:70]!r}')
    bak = path + f'.bak_wpwr_{STAMP}'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    open(path, 'w', encoding='utf-8').write(src.replace(anchor, insert, 1))
    print(f'  OK    {os.path.basename(path)}: patched (backup {os.path.basename(bak)})')
    return True


# ------------------------------------------------------------------ paper2_env.py
env_p = os.path.join(CODE, 'paper2_env.py')
A1 = "W_REL, RLF_PEN, W_PWR, W_SPEC, W_HO = 1.0, 20.0, 0.15, 0.05, 0.05"
I1 = A1 + """
# --- W_PWR override, added 30 Aug 2026 for the reward-weight sensitivity sweep. ---------------
# The transmit-power penalty weight is the one reward term a reviewer can argue produces the
# bimodal convergence structure (a linear power cost against a step-shaped reliability cliff).
# Testing that needs W_PWR varied WITHOUT touching anything else, so it is read here, at module
# scope, where no import order can defeat it. Unset -> 0.15, the published value, unchanged.
_W_PWR_ENV = os.environ.get('P2_W_PWR')
if _W_PWR_ENV is not None:
    W_PWR = float(_W_PWR_ENV)
# ----------------------------------------------------------------------------------------------"""
patch(env_p, A1, I1, '_W_PWR_ENV')

# ------------------------------------------------------------------ paper2_controller.py
ctl_p = os.path.join(CODE, 'paper2_controller.py')

A2 = ("    ap.add_argument('--threads', type=int, default=1, "
      "help='torch CPU threads (1 = deployment condition)')\n"
      "    a = ap.parse_args()\n"
      "    torch.set_num_threads(max(1, a.threads))\n"
      "    if a.p1:\n"
      "        sys.path.insert(0, a.p1)\n"
      "    import pandas as pd, paper2_env as PE\n")
I2 = ("    ap.add_argument('--threads', type=int, default=1, "
      "help='torch CPU threads (1 = deployment condition)')\n"
      "    ap.add_argument('--w_pwr', type=float, default=None,\n"
      "                    help='override the reward transmit-power penalty weight "
      "(paper2_env.W_PWR, default 0.15). '\n"
      "                         'Reward-weight sensitivity sweep, 30 Aug 2026.')\n"
      "    a = ap.parse_args()\n"
      "    torch.set_num_threads(max(1, a.threads))\n"
      "    if a.p1:\n"
      "        sys.path.insert(0, a.p1)\n"
      "    # set BEFORE the env is imported: paper2_env reads P2_W_PWR at module scope\n"
      "    if a.w_pwr is not None:\n"
      "        os.environ['P2_W_PWR'] = repr(float(a.w_pwr))\n"
      "    import pandas as pd, paper2_env as PE\n"
      "    # G1: fail in the first second, not after six minutes of training on the wrong reward\n"
      "    if a.w_pwr is not None and abs(PE.W_PWR - a.w_pwr) > 1e-12:\n"
      "        sys.exit(f'ABORT: asked for W_PWR={a.w_pwr!r} but paper2_env.W_PWR is "
      "{PE.W_PWR!r}. '\n"
      "                 'The override did not take -- refusing to train.')\n")
patch(ctl_p, A2, I2, "--w_pwr")

# G2: the config banner
A3 = ('    print(f"train scenes={len(env.scene_ids)} held-out scenes={len(env_te.scene_ids)} | "')
I3 = ('    print(f"reward weights: W_REL={PE.W_REL} RLF_PEN={PE.RLF_PEN} W_PWR={PE.W_PWR} "\n'
      '          f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO}"'
      ' + ("   <-- W_PWR OVERRIDDEN" if a.w_pwr is not None else ""))\n'
      '    print(f"train scenes={len(env.scene_ids)} held-out scenes={len(env_te.scene_ids)} | "')
patch(ctl_p, A3, I3, 'reward weights: W_REL=')

# G3: the metrics column
A4 = ("    m = dict(seed=a.seed, alpha=a.alpha, lstm=int(not a.ff), "
      "anticipation=int(not a.no_ant),\n"
      "             per=int(not a.no_per), episodes=a.episodes, "
      "train_s=round(time.time() - t0, 1))")
I4 = ("    m = dict(seed=a.seed, alpha=a.alpha, lstm=int(not a.ff), "
      "anticipation=int(not a.no_ant),\n"
      "             per=int(not a.no_per), episodes=a.episodes, w_pwr=float(PE.W_PWR),\n"
      "             train_s=round(time.time() - t0, 1))")
patch(ctl_p, A4, I4, 'w_pwr=float(PE.W_PWR)')

print('\npatch complete.')
