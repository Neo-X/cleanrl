"""Shared utilities for the discrete-action offline interventions.

Loads a dataset saved by ``cleanrl/offline_dataset.py`` (D4RL-style npz), splits it into
trajectories, rebuilds the online agent's network and environment, evaluates policies,
and computes the reference values for the exploitation gap:

    ref/topk_D        V^{pi*_D}(s0): mean return of the top-k% *complete* episodes in D
    ref/mean_D        mean return of complete episodes in D (behaviour data, incl. epsilon)
    ref/online_det    online checkpoint, deterministic (argmax Q), same eval protocol
    eval/gap_closed   (V(offline) - ref/online_det) / (ref/topk_D - ref/online_det)

Algorithms are adapted from CORL (https://github.com/corl-team/CORL, Apache-2.0) to
discrete actions; see README.md in this folder.
"""
import json
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_HERE))                   # cleanrl/
sys.path.append(os.path.dirname(os.path.dirname(_HERE)))  # repo root


# ----------------------------------------------------------------------------- data
def load_dataset(path):
    d = np.load(path)
    data = {k: d[k] for k in d.files}
    model_path = path.replace(".npz", "_model.pt")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False) if os.path.exists(model_path) else None
    data["meta"] = ckpt["meta"] if ckpt else {}
    data["online_state_dict"] = ckpt["state_dict"] if ckpt else None
    data["trajectories"] = split_trajectories(data)
    return data


def split_trajectories(data):
    """Return a list of dicts {start, end (exclusive), ret, complete}.

    A trajectory is complete if it ends in a terminal and did not start before the
    buffer's first stored transition (the circular buffer may have overwritten it)."""
    N = len(data["rewards"])
    n_envs = int(data.get("n_envs", 1))
    T = N // n_envs
    ends = np.flatnonzero(data["terminals"] | data["timeouts"])
    starts_mid = bool(data.get("starts_mid_episode", False))
    trajs, s = [], 0
    for e in ends:
        e = int(e) + 1
        chunk_start = (s // T) * T
        complete = bool(data["terminals"][e - 1]) and not (starts_mid and s == chunk_start)
        trajs.append(dict(start=s, end=e, ret=float(data["rewards"][s:e].sum()), complete=complete))
        s = e
    return trajs


def complete_returns(data):
    return np.array([t["ret"] for t in data["trajectories"] if t["complete"]], dtype=np.float64)


def topk_mean(returns, k):
    if len(returns) == 0:
        return float("nan")
    n = max(int(np.floor(k * len(returns))), 1)
    return float(np.mean(np.sort(returns)[::-1][:n]))


def reference_values(data, k=0.05):
    rets = complete_returns(data)
    online = data.get("online_returns", np.array([]))
    return {
        "ref/topk_D": topk_mean(rets, k),
        "ref/mean_D": float(rets.mean()) if len(rets) else float("nan"),
        "ref/max_D": float(rets.max()) if len(rets) else float("nan"),
        "ref/num_complete_episodes_D": int(len(rets)),
        # the window BufferGapV2 used online (last `return_buffer_size` episodes)
        "ref/topk_online_window": topk_mean(online, k),
        "ref/mean_online_window": float(online.mean()) if len(online) else float("nan"),
    }


class TransitionSampler:
    """Uniform transition sampler, optionally oversampling transitions from the
    top-`top_frac` complete trajectories with probability `top_prob` per sample."""

    def __init__(self, data, device, top_frac=0.0, top_prob=0.0):
        self.device = device
        self.obs = torch.as_tensor(data["observations"]).to(device)
        self.next_obs = torch.as_tensor(data["next_observations"]).to(device)
        self.actions = torch.as_tensor(data["actions"]).long().to(device)
        self.rewards = torch.as_tensor(data["rewards"]).float().to(device)
        self.terminals = torch.as_tensor(data["terminals"]).float().to(device)
        self.N = len(data["rewards"])
        self.top_prob = top_prob
        self.top_idx = None
        if top_frac > 0 and top_prob > 0:
            self.top_idx = top_trajectory_indices(data, top_frac)
            print(f"[sampler] oversampling {len(self.top_idx)} transitions from top {top_frac:.0%} trajectories "
                  f"with prob {top_prob}")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.N, size=batch_size)
        if self.top_idx is not None:
            use_top = np.random.rand(batch_size) < self.top_prob
            idx[use_top] = self.top_idx[np.random.randint(0, len(self.top_idx), size=use_top.sum())]
        idx = torch.as_tensor(idx, device=self.device)
        return (self.obs[idx].float(), self.actions[idx], self.rewards[idx],
                self.next_obs[idx].float(), self.terminals[idx])


def top_trajectory_indices(data, frac):
    """Transition indices of the top-`frac` complete trajectories by undiscounted return
    (CORL's keep_best_trajectories, restricted to complete episodes)."""
    trajs = sorted([t for t in data["trajectories"] if t["complete"]], key=lambda t: -t["ret"])
    keep = trajs[: max(1, int(frac * len(trajs)))]
    return np.concatenate([np.arange(t["start"], t["end"]) for t in keep])


# ----------------------------------------------------------------------------- networks
class QNetwork(nn.Module):
    """Same architecture (and state_dict keys) as cleanrl/dqn.py:QNetwork, so the online
    checkpoint can be loaded. Also used as the logits network for BC."""

    def __init__(self, obs_shape, n_actions, num_layers=1, num_units=120, use_layer_norm=False):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(int(np.prod(obs_shape)), num_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_units, num_units), nn.ReLU()]
            if use_layer_norm:
                layers.append(nn.LayerNorm(num_units))
        layers += [nn.Linear(num_units, 84), nn.ReLU(), nn.Linear(84, n_actions)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def online_net_kwargs(data):
    a = data["meta"].get("args", {})
    return dict(num_layers=a.get("num_layers", 1), num_units=a.get("num_units", 120),
                use_layer_norm=a.get("use_layer_norm", False))


# ----------------------------------------------------------------------------- env / eval
def env_id_of(data, override=""):
    return override or data["meta"]["args"]["env_id"]


def make_env(env_id):
    # Must match the data-generating env in cleanrl/dqn.py:make_env (plain gym.make).
    return gym.make(env_id)


@torch.no_grad()
def evaluate(act_fn, env_id, n_episodes, seed, max_steps=10_000, reset_fn=None):
    """Run `n_episodes` with act_fn(obs) -> int. `reset_fn()` is called at each episode
    start (used by DT to clear its context). Episodes are cut at `max_steps`."""
    env = make_env(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if reset_fn is not None:
            reset_fn()
        ret, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = act_fn(obs)
            obs, r, term, trunc, _ = env.step(a)
            if hasattr(act_fn, "observe"):
                act_fn.observe(a, r, obs)
            ret += r
            t += 1
            done = term or trunc
        returns.append(ret)
    env.close()
    return np.array(returns)


def q_act_fn(net, device):
    def act(obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return int(net(x).argmax(1).item())
    return act


def online_checkpoint_return(data, env_id, n_episodes, seed, max_steps, device):
    """Deterministic (argmax Q) return of the online agent at this checkpoint, using the
    same evaluation protocol as the offline policies."""
    if data["online_state_dict"] is None:
        return float("nan")
    env = make_env(env_id)
    net = QNetwork(env.observation_space.shape, env.action_space.n, **online_net_kwargs(data)).to(device)
    net.load_state_dict(data["online_state_dict"])
    net.eval()
    return float(evaluate(q_act_fn(net, device), env_id, n_episodes, seed, max_steps).mean())


# ----------------------------------------------------------------------------- logging
def set_seed(seed, deterministic_torch=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic_torch


class Logger:
    def __init__(self, args, algo, dataset_path):
        ds = os.path.splitext(os.path.basename(dataset_path))[0]
        src = os.path.basename(os.path.dirname(os.path.abspath(dataset_path)))
        self.run_name = f"{algo}__{src}__{ds}__{args.seed}__{int(time.time())}"
        log_dir = getattr(args, "log_dir", "") or ""
        if getattr(args, "track", False):
            import wandb
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                       config=vars(args), name=self.run_name, group=f"offline-{algo}", save_code=True,
                       dir=log_dir or None)
        self.writer = SummaryWriter(f"{log_dir}runs/offline/{self.run_name}")
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))
        self.results_path = f"{log_dir}runs/offline/{self.run_name}/results.json"
        self.history = []

    def log(self, d, step):
        for k, v in d.items():
            if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                self.writer.add_scalar(k, v, step)

    def log_eval(self, step, returns, refs, extra=None):
        mean = float(np.mean(returns))
        d = {"eval/return": mean, "eval/return_std": float(np.std(returns)), **refs}
        base, top = refs.get("ref/online_det", np.nan), refs.get("ref/topk_D", np.nan)
        if np.isfinite(base) and np.isfinite(top) and top != base:
            d["eval/gap_closed"] = (mean - base) / (top - base)
        if extra:
            d.update(extra)
        self.log(d, step)
        self.history.append({"step": step, **{k: float(v) for k, v in d.items()}})
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(self.history, f, indent=1)
        print(f"step={step} eval/return={mean:.2f}  online_det={base:.2f}  topk_D={top:.2f}  "
              f"gap_closed={d.get('eval/gap_closed', float('nan')):.3f}")

    def close(self):
        self.writer.close()


def compute_refs(data, args, env_id, device):
    refs = reference_values(data, args.top_k)
    refs["ref/online_det"] = online_checkpoint_return(data, env_id, args.eval_episodes, args.eval_seed,
                                                      args.eval_max_steps, device)
    print("[refs] " + ", ".join(f"{k}={v:.2f}" for k, v in refs.items()))
    return refs
