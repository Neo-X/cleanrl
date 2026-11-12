"""Fig. 1(b) regenerated from collect experiment 38 (DQN, MinAtar SpaceInvaders-v0, 5 seeds), with the
offline DT/CQL results (offline_runs.csv, from make_offline_table.py) as markers at the dataset checkpoints.

    uv run python data/iclr2026/make_offline_figure.py            # uses cached dqn_exp57_history.csv if present
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
HIST = os.path.join(HERE, "dqn_exp57_history.csv")
HIST_CQL = os.path.join(HERE, "dqn_cql_exp58_history.csv")  # online CQL (cleanrl/dqn_cql.py), same settings
KEYS = {
    "charts/episodic_return": '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (avg)',
    "charts/deterministic_returns": '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (deterministic)',
    "charts/avg_top_returns_global": '$V^{ \\hat{\\pi}^{*}_{ D_{\\infty} } }(s_0)$ (best)',
    "charts/avg_top_returns_local": '$V^{ \\hat{\\pi}^{*}_{D} }(s_0)$ (recent)',
    "charts/replay_best_returns": '$V^{ \\hat{\\pi}^{*} }(s_0)$ (replay)',
}
BIN = 500_000  # steps per bin for the curves

def fetch(prefix, path, keys):
    if "--refetch" not in sys.argv and os.path.exists(path):
        return pd.read_csv(path)
    import wandb
    frames = []
    for r in wandb.Api().runs("glen-berseth/sub-optimality", filters={"display_name": {"$regex": "^" + prefix}}):
        for k in keys:
            h = pd.DataFrame(r.scan_history(keys=["global_step", k], page_size=10000))
            if len(h):
                frames.append(pd.DataFrame(dict(seed=int(r.name.split("__")[3]), key=k,
                                                step=h["global_step"], value=h[k])))
        print("fetched", r.name)
    df = pd.concat(frames)
    df.to_csv(path, index=False)
    return df


hist = fetch("MinAtar/SpaceInvaders-v0__dqn__57__", HIST, KEYS)
hist_cql = fetch("MinAtar/SpaceInvaders-v0__dqn_cql__58__", HIST_CQL, KEYS)
hist_cql["step"] = (hist_cql["step"] // BIN) * BIN + BIN / 2
hist["step"] = (hist["step"] // BIN) * BIN + BIN / 2

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for k, label in KEYS.items():
    d = hist[hist.key == k].groupby(["seed", "step"], as_index=False)["value"].mean()
    sns.lineplot(data=d, x="step", y="value", ax=ax, label=label, color=colors[label], linewidth=3,
                 errorbar="se")
    ax.lines[-1].set_linestyle(linestyle[label])

# offline interventions: mean over 3 offline seeds per dataset, then mean ± std over the 5 collect seeds
off = pd.read_csv(os.path.join(HERE, "offline_runs.csv"))
for algo, label, color, marker, dx in [("dt", "DT (offline, same data)", "#1b9e77", "D", -4e5),
                                       ("cql", "CQL (offline, same data)", "#d95f02", "s", 4e5)]:
    g = off[off.algo == algo].groupby(["step", "collect_seed"])["eval/return"].mean().groupby("step")
    m, s = g.mean(), g.std(ddof=0)
    ax.errorbar(m.index + dx, m.values, yerr=s.values, fmt=marker, color=color, markersize=10,
                capsize=5, elinewidth=2, markeredgecolor="black", label=label, zorder=5)

ax.set_ylim(0, None)  # linear axis covering every point (deterministic evaluations reach >10k)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
ax.set(ylabel="Return", xlabel="Steps", title="Return on MinAtar Space Invaders (DQN)")
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout(pad=0.5)
out = os.path.join(HERE, "Return_on_MinAtar_Space_Invaders_offline")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}")
print("saved", out + ".{png,pdf,svg}")


# ---------------------------------------------------------------- second figure: the same plot for online CQL (exp 46)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for k, label in KEYS.items():
    d = hist_cql[hist_cql.key == k].groupby(["seed", "step"], as_index=False)["value"].mean()
    sns.lineplot(data=d, x="step", y="value", ax=ax, label=label, color=colors[label], linewidth=3, errorbar="se")
    ax.lines[-1].set_linestyle(linestyle[label])
ax.set_ylim(0, None)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
ax.set(ylabel="Return", xlabel="Steps", title="Return on MinAtar Space Invaders (online CQL)")
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout(pad=0.5)
out = os.path.join(HERE, "Return_on_MinAtar_Space_Invaders_online_cql")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}")
print("saved", out + ".{png,pdf,svg}")
