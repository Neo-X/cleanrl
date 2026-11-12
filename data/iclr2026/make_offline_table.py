"""Fetch the offline DT/CQL runs (collect experiment 38, MinAtar SpaceInvaders-v0) from wandb,
save every run's summary to offline_runs.csv, and write the rebuttal table to offline_table.md."""
import os
import numpy as np
import pandas as pd
import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
api = wandb.Api()
rows = []
for r in api.runs("glen-berseth/sub-optimality",
                  filters={"display_name": {"$regex": "^(dt|cql)__SpaceInvaders-v0__dqn__38"}}):
    n = r.name.split("__")
    s = {k: v for k, v in r.summary.items() if isinstance(v, (int, float)) and k.split("/")[0] in ("eval", "ref")}
    rows.append(dict(algo=n[0], collect_seed=int(n[4]), step=int(n[6].split("_")[1]), offline_seed=int(n[7]),
                     run_id=r.id, state=r.state, **s))
df = pd.DataFrame(rows).sort_values(["algo", "step", "collect_seed", "offline_seed"])
df.to_csv(os.path.join(HERE, "offline_runs.csv"), index=False)

def ms(x):
    return f"{np.mean(x):.0f} ± {np.std(x):.0f}"


lines = ["| Checkpoint | Online agent, argmax Q | Top-5% of D | DT | CQL |", "|---|---|---|---|---|"]
for step, g in df.groupby("step"):
    ref = g.groupby("collect_seed")[["ref/online_det", "ref/topk_D"]].first()
    per = lambda a: g[g.algo == a].groupby("collect_seed")["eval/return"].mean()
    lines.append(f"| {step/1e6:g}M | {ms(ref['ref/online_det'])} | {ms(ref['ref/topk_D'])} | {ms(per('dt'))} | {ms(per('cql'))} |")
lines += ["", "Mean ± std over 5 collect seeds; each seed's value is the mean over 3 offline seeds. "
          "Returns are single deterministic evaluation episodes (argmax policy, deterministic env)."]
open(os.path.join(HERE, "offline_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), f"\n{len(df)} runs saved")
