"""Fetch the offline DT/CQL runs for one collect experiment from wandb, save every run's summary to
<prefix>runs.csv, and write the rebuttal table to <prefix>table.md.

  python make_offline_table.py                      # SpaceInvaders-v0, exp 38 -> offline_*.
  python make_offline_table.py Breakout-v0 59       # -> offline_Breakout-v0_*.
  python make_offline_table.py Asterix-v5 50        # Atari: dt_atari/cql_atari on dqn_atari data
"""
import os
import sys
import numpy as np
import pandas as pd
import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
pos = [x for x in sys.argv[1:] if not x.startswith("--")]
ENV, EXP = (pos[0], pos[1]) if len(pos) > 1 else ("SpaceInvaders-v0", "38")
# --since=YYYY-MM-DD: only runs created after this date (e.g. the 1000-step evaluation cap reruns)
SINCE = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--since=")), None)
PREFIX = "offline_" if ENV == "SpaceInvaders-v0" else f"offline_{ENV}_"
api = wandb.Api(timeout=60)
rows = []
for r in api.runs("glen-berseth/sub-optimality",
                  filters={"display_name": {"$regex": f"^(dt|cql)(_atari)?__{ENV}__dqn(_atari)?__{EXP}__"},
                           **({"created_at": {"$gt": SINCE}} if SINCE else {})}):
    n = r.name.split("__")
    s = {k: v for k, v in r.summary.items() if isinstance(v, (int, float)) and k.split("/")[0] in ("eval", "ref")}
    rows.append(dict(algo=n[0].split("_")[0], collect_seed=int(n[4]), step=int(n[6].split("_")[1]), offline_seed=int(n[7]),
                     run_id=r.id, state=r.state, seq_len=r.config.get("seq_len"), **s))
df = pd.DataFrame(rows).sort_values(["algo", "step", "collect_seed", "offline_seed"])
# --dt_seq_len=N: DT runs with this context length only (20 = the first runs, 1/2 = the short-context reruns)
DT_SEQ = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--dt_seq_len=")), None)
if DT_SEQ is not None:
    df = df[(df.algo != "dt") | (df.seq_len == DT_SEQ)]
df.to_csv(os.path.join(HERE, PREFIX + "runs.csv"), index=False)

# Replay of the best trajectory generated so far (charts/replay_best_returns), from the collect runs' history
# (cached by make_offline_figure.py); the last value logged at or before each dataset checkpoint. Runs without
# a logged history (Breakout exp 82, logging bug) fall back to the best episode in D (ref/max_D).
HISTORY = {"SpaceInvaders-v0": "dqn_exp38_history.csv", "Asterix-v5": "dqn_atari_exp50_history.csv"}


def replay(step):
    path = os.path.join(HERE, HISTORY.get(ENV, "-"))
    if not os.path.exists(path):
        return None
    h = pd.read_csv(path)
    h = h[(h.key == "charts/replay_best_returns") & (h.step <= step)].sort_values("step")
    return h.groupby("seed")["value"].last()


def ms(x):
    return f"{np.mean(x):.0f} ± {np.std(x):.0f}"


rep_label = "Replay of best trajectory" if ENV in HISTORY else "Best episode in D"
lines = [f"| Checkpoint | Online agent, argmax Q | Top-5% of D | {rep_label} | DT | CQL |", "|---|---|---|---|---|---|"]
for step, g in df.groupby("step"):
    ref = g.groupby("collect_seed")[["ref/online_det", "ref/topk_D", "ref/max_D"]].first()
    rep = replay(step)
    rep = ref["ref/max_D"] if rep is None else rep
    per = lambda a: g[g.algo == a].groupby("collect_seed")["eval/return"].mean()
    lines.append(f"| {step/1e6:g}M | {ms(ref['ref/online_det'])} | {ms(ref['ref/topk_D'])} | {ms(rep)} | {ms(per('dt'))} | {ms(per('cql'))} |")
n_off = df.offline_seed.nunique()
lines += ["", "Mean ± std over 5 collect seeds; " + (f"each seed's value is the mean over {n_off} offline seeds. "
          if n_off > 1 else "one offline seed per collect seed. ") +
          "Returns are single deterministic evaluation episodes (argmax policy, deterministic env)."]
open(os.path.join(HERE, PREFIX + "table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), f"\n{len(df)} runs saved")
