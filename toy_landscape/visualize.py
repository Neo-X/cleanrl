"""Visualise *returns over the state space* for the toy landscape.

This module is deliberately **policy-agnostic**: it knows nothing about how a
policy was trained.  The RL algorithms being studied live in ``cleanrl/`` (PPO,
DQN, SAC, PQN, ...); here we only need callable ``policy_fn(obs) -> action``
functions to roll out, plus the env's analytic reward landscape.

It produces the sketch from the paper: the multimodal ``V(s)`` curve with four
markers,

* ``pi*``          -- global optimum of the landscape (green star),
* ``tau_hat*``      -- experience-optimal: best trajectory the agent ever reached (cyan X),
* ``pi_theta``     -- where the *deterministic* (greedy) policy ends up (black X),
* ``pi_theta_s``   -- where the *stochastic* policy typically operates (grey circle).

:func:`plot_ppo_landscape` is a thin, guarded hook so ``cleanrl/ppo.py`` can emit
this figure during training for ``Toy/Landscape-*`` envs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # works both as a package (toy_landscape.visualize) and as a top-level module
    from .env import LandscapeEnv
except ImportError:
    from env import LandscapeEnv

# colours chosen to echo the hand-drawn sketch
C_LEARNED_DET   = "black"      # pi_theta  deterministic
C_LEARNED_STOCH = "#888888"    # pi_theta  stochastic  (grey)
C_BEST          = "#1f9bd6"    # pi_hat*   (cyan)
C_OPT           = "#2ca02c"    # pi*       (green)

PolicyFn = Callable[[np.ndarray], int]


# ----------------------------------------------------------------------------
# Rollout utilities
# ----------------------------------------------------------------------------
def rollout(env: LandscapeEnv, policy_fn: PolicyFn, seed: int | None = None) -> tuple[list[float], float]:
    """Run one episode; return the visited positions and the episode return."""
    obs, info = env.reset(seed=seed)
    xs, ep_ret, done = [info["x"]], 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy_fn(obs))
        xs.append(info["x"])
        ep_ret += r
        done = term or trunc
    return xs, ep_ret


def collect(
    env_factory: Callable[[], LandscapeEnv],
    policy_fn: PolicyFn,
    n_rollouts: int = 30,
    seed0: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Roll out ``policy_fn`` many times.

    Returns the stacked visited positions, the per-episode returns, and the
    median final position (robust estimate of where the policy settles).
    """
    all_x, returns, finals = [], [], []
    for i in range(n_rollouts):
        env = env_factory()
        xs, ep_ret = rollout(env, policy_fn, seed=seed0 + i)
        all_x.extend(xs)
        returns.append(ep_ret)
        finals.append(xs[-1])
    return np.asarray(all_x), np.asarray(returns), float(np.median(finals))


def best_experience_x(visited_x: np.ndarray, env: LandscapeEnv, frac: float = 0.02) -> float:
    """Mean position of the highest-reward states actually visited (``pi_hat*``)."""
    if len(visited_x) == 0:
        return float("nan")
    rewards = env.reward_fn(visited_x)
    n = max(1, int(frac * len(visited_x)))
    top = np.argsort(rewards)[-n:]
    return float(np.mean(visited_x[top]))


def replay_actions(preset: str, actions) -> np.ndarray:
    """Replay an action sequence in a *noise-free* env to recover its visited states.

    Used to turn ``gap_stats``'s best-trajectory actions into positions so that
    ``pi_hat*`` can be placed on the landscape curve.
    """
    env = LandscapeEnv(preset=preset, start_noise=0.0, move_noise=0.0)
    _, info = env.reset(seed=0)
    xs = [info["x"]]
    for a in np.asarray(actions).ravel():
        _, _, term, trunc, info = env.step(int(a))
        xs.append(info["x"])
        if term or trunc:
            break
    return np.asarray(xs)


# ----------------------------------------------------------------------------
# The sketch: V(s) over the state space with all four policy markers
# ----------------------------------------------------------------------------
def plot_landscape(
    env: LandscapeEnv,
    learned_det_x: float | None = None,    # pi_theta  (deterministic greedy)
    learned_stoch_x: float | None = None,  # pi_theta_s (stochastic)
    best_x: float | None = None,           # pi_hat*
    visited_x: np.ndarray | None = None,   # state-visitation histogram
    ax=None,
    title: str | None = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    xs, rs = env.landscape()
    ax.plot(xs, rs, color="0.2", lw=2.5, zorder=3)

    if visited_x is not None and len(visited_x):
        ax.hist(visited_x, bins=80, range=(0, env.length), density=True,
                color="0.7", alpha=0.40, zorder=1)

    def mark(x, color, label, marker="X", size=190, zorder=5, dx=6, dy=8):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return
        y = float(env.reward_fn(np.array([x]))[0])
        ax.scatter([x], [y], s=size, marker=marker, color=color, zorder=zorder,
                   edgecolors="white", linewidths=1.3, clip_on=False)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=13, color=color, weight="bold", clip_on=False)

    # Draw in increasing zorder so the most "important" markers sit on top.
    # pi* as a large star, drawn first so pi_hat* X overlaps it cleanly when
    # exploration finds the global optimum.
    mark(env.global_optimum_x,  C_OPT,           r"$\pi^*$",
         marker="*", size=420, zorder=4, dx=10, dy=-2)
    mark(best_x,                C_BEST,           r"$\hat{\tau}^*$",
         zorder=5, dx=-30, dy=8)
    mark(learned_stoch_x,       C_LEARNED_STOCH,  r"$\pi^\theta_{\rm s}$",
         marker="o", size=180, zorder=6, dx=6, dy=8)
    mark(learned_det_x,         C_LEARNED_DET,    r"$\pi^\theta$",
         zorder=7, dx=6, dy=-16)

    ax.set_xlabel("state  $s$")
    ax.set_ylabel("$V(s)$")
    if title:
        ax.set_title(title)
    ax.margins(x=0.03)
    return ax


def figure_for_policies(
    env_factory: Callable[[], LandscapeEnv],
    det_policy: PolicyFn,
    stoch_policy: PolicyFn | None = None,
    best_x: float | None = None,
    title: str | None = None,
    n_rollouts: int = 30,
):
    """Build the landscape figure given deterministic and (optionally) stochastic policies."""
    env = env_factory()

    # Stochastic rollouts also provide the visitation histogram and best_x estimate.
    ref_policy = stoch_policy if stoch_policy is not None else det_policy
    visited, _, stoch_x = collect(env_factory, ref_policy, n_rollouts=n_rollouts)

    if best_x is None:
        best_x = best_experience_x(visited, env)

    _, _, det_x = collect(env_factory, det_policy, n_rollouts=n_rollouts)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_landscape(
        env,
        learned_det_x=det_x,
        learned_stoch_x=stoch_x if stoch_policy is not None else None,
        best_x=best_x,
        visited_x=visited,
        ax=ax,
        title=title,
    )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Hook for cleanrl/ppo.py (guarded: no-op unless it's a Toy/Landscape env)
# ----------------------------------------------------------------------------
def plot_ppo_landscape(writer, global_step, args, agent, gap_stats, device, save_dir=None):
    """Render the four-marker state-space sketch for the live PPO agent.

    Safe to call unconditionally from ppo.py: it returns immediately unless
    ``args.env_id`` is a ``Toy/Landscape-*`` env, and never raises into training.
    """
    try:
        env_id = getattr(args, "env_id", "")
        if not env_id.startswith("Toy/Landscape"):
            return
        import os
        import re
        import torch

        m = re.match(r"Toy/Landscape-(.+)-v0", env_id)
        preset = m.group(1) if m else "three_peaks"

        @torch.no_grad()
        def det_policy(obs: np.ndarray) -> int:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(agent.get_action_deterministic(t).item())

        # Stochastic policy: PPO agents expose get_action_and_value; DQN-style
        # agents (QNetwork) only have get_action_deterministic, so fall back to it.
        @torch.no_grad()
        def stoch_policy(obs: np.ndarray) -> int:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if hasattr(agent, "get_action_and_value"):
                action, _, _, _ = agent.get_action_and_value(t)
                return int(action.item())
            return int(agent.get_action_deterministic(t).item())

        # pi_hat*: replay the best trajectory gap_stats has stored, if available.
        best_x = None
        best_traj = getattr(gap_stats, "_best_traj", None)
        if best_traj is not None and len(np.asarray(best_traj).ravel()) > 0:
            states = replay_actions(preset, best_traj)
            env_tmp = LandscapeEnv(preset=preset)
            best_x = float(states[int(np.argmax(env_tmp.reward_fn(states)))])

        fig = figure_for_policies(
            lambda: LandscapeEnv(preset=preset),
            det_policy=det_policy,
            stoch_policy=stoch_policy,
            best_x=best_x,
            title=f"PPO on {env_id}  (step {global_step})",
        )
        if writer is not None:
            writer.add_figure("landscape/state_value", fig, global_step)
        save_dir = save_dir or getattr(args, "log_dir", "runs/")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"landscape_{preset}_{global_step}.png"), dpi=120)
        plt.close(fig)
    except Exception as e:  # never break training because of a plot
        print(f"[toy_landscape] landscape plot skipped: {e}")
