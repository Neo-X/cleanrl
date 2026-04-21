import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.2)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_plots import add_plot, colors, get_jobs, render_plot


REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

CORE_PPO_FILES = {
    "PPO_HalfCheetah-v4.csv",
    "PPO_Humanoid-v4.csv",
    "PPO_Walker2d.csv",
    "PPO_Crafter.csv",
    "PPO_MinAtar_SpaceInvaders.csv",
}

PLOT_SERIES = [
    (" - charts/replay_best_returns", '$V^{ \\hat{\\pi}^{*} }(s_0)$ (replay)'),
    (" - charts/episodic_return", '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (avg)'),
    (" - charts/avg_top_returns_global", '$V^{ \\hat{\\pi}^{*}_{ D_{\\infty} } }(s_0)$ (best)'),
    (" - charts/avg_top_returns_local", '$V^{ \\hat{\\pi}^{*}_{D} }(s_0)$ (recent)'),
    (" - charts/replay_top_k_returns_stochastic", '$V^{ \\hat{\\pi}^{*} }(s_0)$ (top replay)'),
    (" - charts/deterministic_returns", '$V^{ \\hat{\\pi}^{\\theta} }(s_0)$ (deterministic)'),
]


def iter_ppo_data_files():
    files = []
    for path in sorted(DATA_DIR.glob("PPO*.csv")):
        if path.name.startswith("PPO_on_") or path.name in CORE_PPO_FILES:
            files.append(path)
    return files


def has_required_metrics(df, jobs):
    if not jobs:
        return False

    for metric_key, _ in PLOT_SERIES:
        for job in jobs:
            if f"{job}{metric_key}" not in df.columns:
                return False
    return True


if __name__ == '__main__':
    os.chdir(REPO_ROOT)

    res = 10
    lw_ = 3

    for datadir in iter_ppo_data_files():
        df = pd.read_csv(datadir)
        jobs = get_jobs(df)
        if not has_required_metrics(df, jobs):
            print(f"Skipping {datadir.name}: missing expected PPO plotting metrics.")
            continue

        title = datadir.stem
        print(f"Plotting {title}")

        fig, ax3 = plt.subplots(1, 1, figsize=(8, 4.5))
        ax3.set_title(title)

        for metric_key, label in PLOT_SERIES:
            add_plot(
                ax3,
                df,
                key=metric_key,
                label=label,
                res=res,
                jobs=jobs,
                color=colors[label],
                lw=lw_,
            )

        render_plot(ax3, fig, title.replace('_', ' '))
        fig.savefig("data/"+title+".svg")
        fig.savefig("data/"+title+".png")
        fig.savefig("data/"+title+".pdf")