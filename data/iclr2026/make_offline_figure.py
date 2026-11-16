"""Per-env return figures (DQN curves, 5 seeds) with the offline DT/CQL results (from make_offline_table.py)
as markers at the dataset checkpoints.

    uv run python data/iclr2026/make_offline_figure.py                 # every env in ENVS; cached histories if present
    uv run python data/iclr2026/make_offline_figure.py Breakout-v0     # one env
    uv run python data/iclr2026/make_offline_figure.py --refetch
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "plotting"))
from make_plots import colors, linestyle  # noqa: E402  (paper's colours/line styles)

sns.set(font_scale=1.2)
# env -> (title, DQN run-name prefix, history csv, offline runs csv, collect runs to keep (None = all))
ENVS = {
    "SpaceInvaders-v0": ("MinAtar Space Invaders", "MinAtar/SpaceInvaders-v0__dqn__57__", "dqn_exp57_history.csv",
                         "offline_runs.csv", None),
    # Breakout: see make_breakout_figure.py (the paper's DQN curves; the reruns lack return logs)
    # Asterix: the 5 collect runs (exp 50) whose 12.5M/25M datasets were used; seed 1 has extra restarted runs
    "Asterix-v5": ("Atari Asterix", "ALE/Asterix-v5__dqn_atari__50__", "dqn_atari_exp50_history.csv",
                   "offline_Asterix-v5_runs.csv", ("1794369076", "1794369078")),
}
KEYS = {
    "charts/episodic_return": '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (avg)',
    "charts/deterministic_returns": '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (deterministic)',
    "charts/avg_top_returns_global": '$V^{ \\hat{\\pi}^{*}_{ D_{\\infty} } }(s_0)$ (best)',
    "charts/avg_top_returns_local": '$V^{ \\hat{\\pi}^{*}_{D} }(s_0)$ (recent)',
    "charts/replay_best_returns": '$V^{ \\hat{\\pi}^{*} }(s_0)$ (replay)',
}
BIN = 500_000  # steps per bin for the curves

def fetch(prefix, path, keys=KEYS, keep=None):
    if "--refetch" not in sys.argv and os.path.exists(path):
        return pd.read_csv(path)
    import wandb
    frames = []
    for r in wandb.Api(timeout=60).runs("glen-berseth/sub-optimality", filters={"display_name": {"$regex": "^" + prefix}}):
        if keep and r.name.split("__")[-1] not in keep:
            continue
        for k in keys:
            h = pd.DataFrame(r.scan_history(keys=["global_step", k], page_size=10000))
            if len(h):
                frames.append(pd.DataFrame(dict(seed=int(r.name.split("__")[3]), key=k,
                                                step=h["global_step"], value=h[k])))
        print("fetched", r.name)
    df = pd.concat(frames)
    df.to_csv(path, index=False)
    return df


def make_figures(env, title, prefix, hist_csv, runs_csv, keep):
    hist = fetch(prefix, os.path.join(HERE, hist_csv), keep=keep)
    hist["step"] = (hist["step"] // BIN) * BIN + BIN / 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    atari = env.endswith("-v5")
    for k, label in KEYS.items():
        if atari and k == "charts/episodic_return":
            continue  # raw game score; every other curve and the offline evaluations are in clipped-reward units
        d = hist[hist.key == k].groupby(["seed", "step"], as_index=False)["value"].mean()
        sns.lineplot(data=d, x="step", y="value", ax=ax, label=label, color=colors[label], linewidth=3,
                     errorbar="se")
        ax.lines[-1].set_linestyle(linestyle[label])

    # offline interventions: mean over the offline seeds per dataset, then mean ± std over the 5 collect seeds
    off = pd.read_csv(os.path.join(HERE, runs_csv))
    for algo, label, color, marker, dx in [("dt", "DT (offline, same data)", "#1b9e77", "D", -4e5),
                                           ("cql", "CQL (offline, same data)", "#d95f02", "s", 4e5)]:
        g = off[off.algo == algo].groupby(["step", "collect_seed"])["eval/return"].mean().groupby("step")
        m, s = g.mean(), g.std(ddof=0)
        ax.errorbar(m.index + dx, m.values, yerr=s.values, fmt=marker, color=color, markersize=10,
                    capsize=5, elinewidth=2, markeredgecolor="black", label=label, zorder=5)

    ax.set_ylim(0, None)  # linear axis covering every point
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
    ax.set(ylabel="Return (clipped rewards)" if atari else "Return", xlabel="Steps", title=f"Return on {title} (DQN)")
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout(pad=0.5)
    out = os.path.join(HERE, f"Return_on_{title.replace(' ', '_')}_offline")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}")
    print("saved", out + ".{png,pdf,svg}")



for env in [a for a in sys.argv[1:] if not a.startswith("--")] or ENVS:
    make_figures(env, *ENVS[env])
