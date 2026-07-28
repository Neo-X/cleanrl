"""Compute the variance-target k* / alpha* diagnostic from existing wandb CSV
exports and summarize it (per-seed, per-environment, and per-algorithm) into a
table -- no new training runs required.

Two numbers are reported per run, where available:
  - logged_k_star / logged_alpha_star: the last value of charts/k_star and
    charts/alpha_star that buffer_gap logged *during training* (only present
    for runs launched after that logging was added; NaN otherwise).
  - recomputed_k_star / recomputed_alpha_star: k* recomputed from scratch
    with buffer_gap.compute_optimal_k over *all* episodic returns collected
    over the entire run (charts/episodic_return). This works for any CSV
    export, old or new, since it only needs charts/episodic_return.

Follows the codebase's existing convention of one CSV per algorithm+environment
(e.g. data/DQN_on_ALE_BattleZone.csv, data/PPO_on_ALE_BattleZone.csv).

Usage:
    uv run --with tensorboard python plotting/compute_k_star_table.py --epsilon 0.10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from make_plots import deNan, get_jobs  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cleanrl"))
import buffer_gap  # noqa: E402

# One entry per algorithm; each maps to the per-environment wandb CSV exports
# already sitting in data/ (same files used by the other plotting scripts).
ALGO_CSV_FILES = {
    "DQN": [
        "DQN_on_ALE_BattleZone.csv",
        "DQN_on_ALE_MontezumaRevenge.csv",
        "DQN_on_ALE_NameThisGame.csv",
        "DQN_on_ALE_Asterix.csv",
        "DQN_on_ALE_SpaceInvaders.csv",
    ],
    "PPO": [
        "PPO_on_ALE_BattleZone.csv",
        "PPO_on_ALE_MontezumaRevenge.csv",
        "PPO_on_ALE_NameThisGame.csv",
        "PPO_on_ALE_Asterix.csv",
        "PPO_on_ALE_SpaceInvaders.csv",
    ],
    "PQN": [
        "PQN_on_ALE_MontezumaRevenge.csv",
        "PQN_on_ALE_BattleZone.csv",
        "PQN_on_ALE_Asterix.csv",
        "PQN_on_ALE_NameThisGame.csv",
        "PQN_on_ALE_SpaceInvaders.csv",
    ],
}


def env_from_job(job):
    # e.g. "ALE/BattleZone-v5__dqn_atari__8307128__4__1765852172" -> "ALE/BattleZone-v5"
    return job.split("__")[0]


def last_valid(df, key):
    """Last value of a wandb column, forward/back-filled with make_plots.deNan
    the same way the rest of the plotting pipeline treats missing steps."""
    data_ = deNan(df[key].to_numpy(dtype=float).copy())
    return float(data_[-1])


def recompute_k_star(df, job, epsilon, method):
    # NOTE: unlike last_valid()/get_data_frame(), we deliberately do *not* run this
    # column through deNan() here: deNan forward-fills every non-logging step with the
    # previous episode's return, which would inject many duplicate "samples" into the
    # return distribution and bias the variance estimate. We want only the real,
    # actually-logged per-episode returns.
    key = f"{job} - charts/episodic_return"
    returns = df[key].dropna().to_numpy(dtype=float)
    if len(returns) < 2:
        return np.nan, np.nan
    result = buffer_gap.compute_optimal_k(returns, epsilon=epsilon, method=method)
    return result["k_star"], result["alpha_star"]


def rows_for_csv(csv_path, algo, epsilon, method):
    df = pd.read_csv(csv_path)
    # Use charts/episodic_return (present in every export) to discover jobs,
    # since older runs don't have charts/k_star logged at all.
    jobs = get_jobs(df, tag=" - charts/episodic_return")

    rows = []
    for job in jobs:
        has_logged = f"{job} - charts/k_star" in df.columns
        logged_k = last_valid(df, f"{job} - charts/k_star") if has_logged else np.nan
        logged_alpha = last_valid(df, f"{job} - charts/alpha_star") if has_logged else np.nan
        recomputed_k, recomputed_alpha = recompute_k_star(df, job, epsilon, method)
        rows.append(
            {
                "csv": os.path.basename(csv_path),
                "algo": algo,
                "env": env_from_job(job),
                "job": job,
                "logged_k_star": logged_k,
                "logged_alpha_star": logged_alpha,
                "recomputed_k_star": recomputed_k,
                "recomputed_alpha_star": recomputed_alpha,
            }
        )
    return rows


def build_tables(csv_files_by_algo, epsilon=0.10, method="closed_form"):
    rows = []
    for algo, csv_paths in csv_files_by_algo.items():
        for csv_path in csv_paths:
            rows.extend(rows_for_csv(csv_path, algo, epsilon, method))
    per_run = pd.DataFrame(rows)

    agg_kwargs = dict(
        n_seeds=("job", "count"),
        recomputed_k_star_mean=("recomputed_k_star", "mean"),
        recomputed_k_star_std=("recomputed_k_star", "std"),
        recomputed_alpha_star_mean=("recomputed_alpha_star", "mean"),
        recomputed_alpha_star_std=("recomputed_alpha_star", "std"),
        logged_k_star_mean=("logged_k_star", "mean"),
        logged_alpha_star_mean=("logged_alpha_star", "mean"),
    )

    by_algo_env = per_run.groupby(["algo", "env"]).agg(**agg_kwargs).reset_index()
    by_algo = per_run.groupby("algo").agg(**agg_kwargs).reset_index()

    return per_run, by_algo_env, by_algo


def to_markdown_table(df, float_fmt="{:.4g}"):
    """Render a DataFrame as a GitHub-flavored markdown table without depending
    on the optional `tabulate` package (not installed in this uv environment)."""
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else float_fmt.format(v)
        return str(v)

    header = list(df.columns)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in header) + " |")
    return "\n".join(lines)


def to_latex_rows(by_algo):
    lines = []
    for _, row in by_algo.iterrows():
        lines.append(
            f"{row['algo']} & "
            f"{row['recomputed_k_star_mean']:.1f} $\\pm$ {row['recomputed_k_star_std']:.1f} & "
            f"{row['recomputed_alpha_star_mean']:.3f} $\\pm$ {row['recomputed_alpha_star_std']:.3f} \\\\"
        )
    return lines


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--method", choices=["closed_form", "bootstrap"], default="closed_form")
    parser.add_argument("--out-dir", default=data_dir)
    args = parser.parse_args()

    csv_files_by_algo = {
        algo: [os.path.join(data_dir, fname) for fname in fnames]
        for algo, fnames in ALGO_CSV_FILES.items()
    }

    per_run, by_algo_env, by_algo = build_tables(csv_files_by_algo, epsilon=args.epsilon, method=args.method)

    pd.set_option("display.width", 200)
    print("Per-run results:")
    print(per_run.to_string(index=False))

    print("\nPer-algorithm, per-environment summary (mean +/- std over seeds):")
    print(by_algo_env.to_string(index=False))

    print("\nPer-algorithm summary (mean +/- std over environments x seeds):")
    print(by_algo.to_string(index=False))

    print("\nPer-run results (markdown):")
    print(to_markdown_table(per_run))

    print("\nPer-algorithm, per-environment summary (markdown):")
    print(to_markdown_table(by_algo_env))

    print("\nPer-algorithm summary (markdown):")
    print(to_markdown_table(by_algo))

    per_run_csv = os.path.join(args.out_dir, "k_star_per_run.csv")
    by_algo_env_csv = os.path.join(args.out_dir, "k_star_by_algo_env.csv")
    by_algo_csv = os.path.join(args.out_dir, "k_star_by_algo.csv")
    per_run.to_csv(per_run_csv, index=False)
    by_algo_env.to_csv(by_algo_env_csv, index=False)
    by_algo.to_csv(by_algo_csv, index=False)
    print(f"\nSaved per-run table to {per_run_csv}")
    print(f"Saved per-algo/env table to {by_algo_env_csv}")
    print(f"Saved per-algo table to {by_algo_csv}")

    per_run_md = os.path.join(args.out_dir, "k_star_per_run.md")
    by_algo_env_md = os.path.join(args.out_dir, "k_star_by_algo_env.md")
    by_algo_md = os.path.join(args.out_dir, "k_star_by_algo.md")
    with open(per_run_md, "w") as f:
        f.write(to_markdown_table(per_run) + "\n")
    with open(by_algo_env_md, "w") as f:
        f.write(to_markdown_table(by_algo_env) + "\n")
    with open(by_algo_md, "w") as f:
        f.write(to_markdown_table(by_algo) + "\n")
    print(f"Saved per-run markdown table to {per_run_md}")
    print(f"Saved per-algo/env markdown table to {by_algo_env_md}")
    print(f"Saved per-algo markdown table to {by_algo_md}")

    print("\nLaTeX table rows (recomputed k*, alpha*, per algorithm):")
    print("\n".join(to_latex_rows(by_algo)))


if __name__ == "__main__":
    main()
