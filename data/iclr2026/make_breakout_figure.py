"""The paper's MinAtar Breakout DQN figure (4 seeds, 25M steps, data/DQN_Minatar_Breakout_all.csv from the
May 2025 runs) with the offline DT/CQL results (offline_Breakout-v0_runs.csv, from make_offline_table.py,
collect experiment 82) as markers at the 12.5M and 25M dataset checkpoints.

    uv run python data/iclr2026/make_breakout_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "plotting"))
from make_plots import colors, linestyle  # noqa: E402  (paper's colours/line styles)

sns.set(font_scale=1.2)
PAPER_CSV = os.path.join(HERE, "..", "DQN_Minatar_Breakout_all.csv")
KEYS = {
    "charts/episodic_return": '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (avg)',
    "charts/deterministic_returns": '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (deterministic)',
    "charts/avg_top_returns_global": '$V^{ \\hat{\\pi}^{*}_{ D_{\\infty} } }(s_0)$ (best)',
    "charts/avg_top_returns_local": '$V^{ \\hat{\\pi}^{*}_{D} }(s_0)$ (recent)',
    "charts/replay_best_returns": '$V^{ \\hat{\\pi}^{*} }(s_0)$ (replay)',
}
BIN = 500_000  # steps per bin for the curves

# wandb CSV export (one column per run and key) -> long format (seed, key, step, value)
wide = pd.read_csv(PAPER_CSV)
runs = sorted({c.split(" - ")[0] for c in wide.columns if " - " in c})
frames = []
for run in runs:
    step = wide[f"{run} - global_step"]
    for k in KEYS:
        d = pd.DataFrame(dict(step=step, value=wide[f"{run} - {k}"])).dropna()
        frames.append(d.assign(seed=int(run.split("__")[3]), key=k))
hist = pd.concat(frames)
hist["step"] = (hist["step"] // BIN) * BIN + BIN / 2

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for k, label in KEYS.items():
    d = hist[hist.key == k].groupby(["seed", "step"], as_index=False)["value"].mean()
    sns.lineplot(data=d, x="step", y="value", ax=ax, label=label, color=colors[label], linewidth=3, errorbar="se")
    ax.lines[-1].set_linestyle(linestyle[label])

# offline interventions: one offline seed per dataset, mean ± std over the 5 collect seeds
off = pd.read_csv(os.path.join(HERE, "offline_Breakout-v0_runs.csv"))
for algo, label, color, marker, dx in [("dt", "DT (offline, same data)", "#1b9e77", "D", -4e5),
                                       ("cql", "CQL (offline, same data)", "#d95f02", "s", 4e5)]:
    g = off[off.algo == algo].groupby(["step", "collect_seed"])["eval/return"].mean().groupby("step")
    m, s = g.mean(), g.std(ddof=0)
    ax.errorbar(m.index + dx, m.values, yerr=s.values, fmt=marker, color=color, markersize=10,
                capsize=5, elinewidth=2, markeredgecolor="black", label=label, zorder=5)

ax.set_ylim(0, None)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
ax.set(ylabel="Return", xlabel="Steps", title="Return on MinAtar Breakout (DQN)")
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout(pad=0.5)
out = os.path.join(HERE, "Return_on_MinAtar_Breakout_offline")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}")
print("saved", out + ".{png,pdf,svg}")
