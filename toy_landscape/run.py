"""Render the toy reward landscape (the ``V(s)`` sketch) for a chosen preset.

The RL algorithms under study live in ``cleanrl/`` -- train one on the env with,
e.g.::

    .venv/bin/python cleanrl/ppo.py --env_id Toy/Landscape-three_peaks-v0 \
        --total_timesteps 500000 --num_envs 4

which (via the hook in ppo.py) drops the state-space figure into the run's
log_dir as training progresses.

This script is just a quick, training-free way to *see the landscape* and sanity
check the visualisation with reference policies (random, always-right oracle):

    .venv/bin/python toy_landscape/run.py --preset three_peaks
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import LandscapeEnv, REWARD_PRESETS
import visualize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="three_peaks", choices=list(REWARD_PRESETS))
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def env_factory() -> LandscapeEnv:
        return LandscapeEnv(preset=args.preset)

    # Reference policies just to exercise the plot (NOT learning algorithms):
    #   - "stay": stays in the starting basin  -> a local-optimum pi_theta
    #   - "right": always walks toward the global optimum -> an oracle pi*
    # Each panel shows both a "deterministic" and a "stochastic" variant so the
    # four markers (pi*, tau_hat*, pi_theta, pi_theta_s) all appear.
    # Here the two variants are just the same callable for illustration; with a
    # real algorithm they would be the greedy vs sampled action.
    policy_pairs = {
        "stay":  (lambda obs: 1, lambda obs: 1),
        "right": (lambda obs: 2, lambda obs: 2),
    }

    fig, axes = plt.subplots(1, len(policy_pairs), figsize=(7 * len(policy_pairs), 4.5), squeeze=False)
    for ax, (name, (det_pol, stoch_pol)) in zip(axes.ravel(), policy_pairs.items()):
        env = env_factory()
        visited, returns, _ = visualize.collect(env_factory, stoch_pol, n_rollouts=30)
        best_x = visualize.best_experience_x(visited, env)
        _, _, det_x   = visualize.collect(env_factory, det_pol,   n_rollouts=30)
        _, _, stoch_x = visualize.collect(env_factory, stoch_pol, n_rollouts=30)
        visualize.plot_landscape(
            env,
            learned_det_x=det_x,
            learned_stoch_x=stoch_x,
            best_x=best_x,
            visited_x=visited,
            ax=ax,
            title=f"reference policy: {name}  (mean return {returns.mean():.1f})",
        )
    fig.suptitle(f"Toy landscape — preset '{args.preset}'", y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"{args.preset}_reference.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    env = env_factory()
    opt_x = env.global_optimum_x
    print(f"preset            : {args.preset}")
    print(f"global optimum    : x = {opt_x:.2f},  R(x*) = {float(env.reward_fn([opt_x])[0]):.2f}/step")
    print(f"saved figure      : {out}")
    print("\nTrain a real algorithm on it with, e.g.:")
    print(f"  .venv/bin/python cleanrl/ppo.py --env_id Toy/Landscape-{args.preset}-v0 "
          f"--total_timesteps 500000 --num_envs 4")


if __name__ == "__main__":
    main()
