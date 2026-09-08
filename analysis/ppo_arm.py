#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ppo_arm.py — the cross-algorithm arm registered in PREREGISTRATION_PPO_20260908.md.

WHAT IS HELD IDENTICAL TO EVERY OTHER LEARNED ARM IN THE PAPER
--------------------------------------------------------------
  * the environment (paper2_env.MultiRSUBlockageEnv), unmodified
  * the reward and its published weights, unmodified
  * the action space MultiDiscrete([2,3,3]) — the published inc3 power actuator
  * the train/held-out scene split
  * the EVALUATION: this file imports `evaluate` and `summarize` from paper2_controller and
    calls them unchanged, so the 500 held-out episodes, the common random numbers at
    EVAL_SEED_BASE = 900,000,000 and the greedy action rule are byte-for-byte the ones used
    by every QR-DQN run in this paper. The PPO agent exposes the same act(obs, hidden, eps)
    signature for that reason and no other.
  * the training budget in episodes, and the seeds

WHAT DIFFERS: the learner, and only the learner. Clipped-surrogate policy gradient with GAE,
a stochastic branched-categorical policy and a scalar value head. No replay buffer, no target
network, no quantile critic, no risk measure.

INTEGRITY: nothing here reads or reacts to an outcome. Hyper-parameters are the standard PPO
defaults recorded below and were fixed before the first run; they are not tuned against the
result. If PPO fails the competence gate the pre-registration names exactly one fallback.
"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.distributions import Categorical

import paper2_env as PE
from paper2_controller import EVAL_SEED_BASE, evaluate, summarize, GAMMA

BRANCHES = [2, 3, 3]

# --- PPO hyper-parameters: standard values, fixed before the first run ------------
CLIP        = 0.2
GAE_LAMBDA  = 0.95
LR          = 3e-4
EPOCHS      = 10        # optimisation epochs per update
MINIBATCH   = 256
ENT_COEF    = 0.01
VF_COEF     = 0.5
MAX_GRAD    = 0.5
EP_PER_UPD  = 8         # episodes collected per policy update


class PPONet(nn.Module):
    """Shared trunk, one categorical head per actuator branch, one scalar value head."""
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.heads = nn.ModuleList([nn.Linear(hidden, nb) for nb in BRANCHES])
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return [head(h) for head in self.heads], self.v(h).squeeze(-1)


class PPOAgent:
    """Same act() signature as paper2_controller.Agent, so evaluate() is reused unchanged.

    `hidden` is accepted and returned as None: PPO here is feed-forward. `eps` selects greedy
    (eps == 0, used by evaluate) versus sampling from the policy (used while collecting).
    """
    def __init__(self, obs_dim, device='cpu'):
        self.net = PPONet(obs_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.device = device

    @torch.no_grad()
    def act(self, obs, hidden, eps):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        logits, _ = self.net(x)
        if eps > 0.0:                       # collection: sample
            a = [int(Categorical(logits=l).sample()) for l in logits]
        else:                               # evaluation: greedy, as for every other arm
            a = [int(l.argmax(dim=-1)) for l in logits]
        return a, None

    @torch.no_grad()
    def act_with_logp(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        logits, v = self.net(x)
        acts, logp = [], 0.0
        for l in logits:
            d = Categorical(logits=l)
            s = d.sample()
            acts.append(int(s)); logp = logp + float(d.log_prob(s))
        return acts, logp, float(v)

    def update(self, obs, acts, logp_old, adv, ret):
        obs = torch.tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        acts = torch.tensor(np.asarray(acts), dtype=torch.long, device=self.device)
        logp_old = torch.tensor(np.asarray(logp_old), dtype=torch.float32, device=self.device)
        adv = torch.tensor(np.asarray(adv), dtype=torch.float32, device=self.device)
        ret = torch.tensor(np.asarray(ret), dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = len(obs)
        idx = np.arange(n)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, n, MINIBATCH):
                b = idx[s:s + MINIBATCH]
                logits, v = self.net(obs[b])
                logp, ent = 0.0, 0.0
                for j, l in enumerate(logits):
                    d = Categorical(logits=l)
                    logp = logp + d.log_prob(acts[b][:, j])
                    ent = ent + d.entropy()
                ratio = torch.exp(logp - logp_old[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[b]
                loss = (-torch.min(s1, s2).mean()
                        + VF_COEF * ((v - ret[b]) ** 2).mean()
                        - ENT_COEF * ent.mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD)
                self.opt.step()


def gae(rews, vals, last_v, terminal):
    """Generalised advantage estimation for one episode.

    `terminal` is True only when the episode ended in sustained radio link failure, a true
    absorbing state. When it ends at the scene time limit the value of the final state is
    bootstrapped instead — the same distinction paper2_controller makes for its n-step target.
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float64)
    nextv, gae_ = (0.0 if terminal else last_v), 0.0
    for t in reversed(range(T)):
        delta = rews[t] + GAMMA * nextv - vals[t]
        gae_ = delta + GAMMA * GAE_LAMBDA * gae_
        adv[t] = gae_
        nextv = vals[t]
    return adv, adv + np.asarray(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    pool = pd.read_csv(a.pool)
    env = PE.MultiRSUBlockageEnv(pool, seed=a.seed, split='train')
    env_te = PE.MultiRSUBlockageEnv(pool, seed=a.seed + 10_000, split='test')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(1)                       # CPU-only, single-threaded, as in every other arm
    agent = PPOAgent(obs_dim=env.observation_space.shape[0])

    print(f'PPO | seed {a.seed} | {a.episodes} episodes | train scenes={len(env.scene_ids)} '
          f'held-out={len(env_te.scene_ids)} | gamma={GAMMA} clip={CLIP} lam={GAE_LAMBDA}', flush=True)
    t0 = time.time()
    hist, buf = [], dict(obs=[], acts=[], logp=[], adv=[], ret=[])
    for ep in range(a.episodes):
        obs, _ = env.reset()
        O, A, L, R, V = [], [], [], [], []
        done = trunc = False
        while not (done or trunc):
            acts, logp, v = agent.act_with_logp(obs)
            O.append(obs); A.append(acts); L.append(logp); V.append(v)
            obs, r, done, trunc, _ = env.step(acts)
            R.append(r)
        last_v = 0.0 if done else agent.act_with_logp(obs)[2]
        adv, ret = gae(R, V, last_v, terminal=bool(done))
        buf['obs'] += O; buf['acts'] += A; buf['logp'] += L
        buf['adv'] += list(adv); buf['ret'] += list(ret)
        hist.append(float(np.sum(R)))
        if (ep + 1) % EP_PER_UPD == 0:
            agent.update(buf['obs'], buf['acts'], buf['logp'], buf['adv'], buf['ret'])
            buf = dict(obs=[], acts=[], logp=[], adv=[], ret=[])
        if (ep + 1) % 250 == 0:
            print(f'  ep {ep+1}/{a.episodes}  train return (last 100) '
                  f'{np.mean(hist[-100:]):.2f}  {time.time()-t0:.0f}s', flush=True)

    te = evaluate(env_te, agent, a.eval_n, seed_base=EVAL_SEED_BASE)
    m = summarize(te)
    m.update(dict(config='ppo', seed=a.seed, learned=1, episodes=a.episodes,
                  wall_s=round(time.time() - t0, 1)))
    os.makedirs(a.out, exist_ok=True)
    te.to_csv(os.path.join(a.out, f'ppo_s{a.seed}_eval.csv'), index=False)
    pd.DataFrame([m]).to_csv(os.path.join(a.out, f'ppo_s{a.seed}_metrics.csv'), index=False)
    pd.DataFrame(dict(episode=np.arange(len(hist)), ret=hist)).to_csv(
        os.path.join(a.out, f'ppo_s{a.seed}_train.csv'), index=False)
    print(f"seed {a.seed}: return={m['return_mean']:.2f} RLF={m['rlf_rate']:.2f}% "
          f"pt={m['pt_dbm']:.4f}dBm rb={m['rb']:.1f} | {m['wall_s']}s", flush=True)


if __name__ == '__main__':
    main()
