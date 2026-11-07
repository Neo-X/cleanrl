"""Launch the fixed-data intervention experiments with xm-slurm on Mila or DRAC (Narval, Rorqual).

The container is built from this repo's Dockerfile, so the local working tree (including
uncommitted changes) is what runs, and pushed to Docker Hub (`IMAGE`). Every cluster mounts
its scratch at /scratch inside the container, so all logs and datasets go to /scratch/:

    Mila:  /network/scratch/g/glen.berseth/   DRAC: /scratch/gberseth/

Run with the xm-slurm environment:

    # 1. online DQN, saving (model, replay buffer) at 25/50/75/100% of training
    ../xm-slurm/.venv/bin/python launch_xm.py collect --cluster narval --env_ids MinAtar/SpaceInvaders-v1

    # 2. every offline intervention on every dataset found on that cluster's scratch
    ../xm-slurm/.venv/bin/python launch_xm.py offline --cluster narval --env_ids MinAtar/SpaceInvaders-v1

    # print the jobs without building or submitting anything
    ../xm-slurm/.venv/bin/python launch_xm.py collect --cluster mila --dry_run
"""
import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys

from absl import app

IMAGE = "gberseth/cleanrl-slurm:latest"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# cluster -> (ssh host alias, user, account, partition, scratch path on the host)
CLUSTERS = {
    "mila": dict(ssh="mila", user="glen.berseth", account=None, partition="long-cpu",
                 scratch="/network/scratch/g/glen.berseth"),
    "narval": dict(ssh="narval", user="gberseth", account="rrg-gberseth", partition=None,
                   scratch="/scratch/gberseth"),
    "rorqual": dict(ssh="rorqual", user="gberseth", account="rrg-gberseth", partition=None,
                    scratch="/scratch/gberseth"),
}

# Offline interventions run on every dataset: (name, script, extra args)
OFFLINE_VARIANTS = [
    ("bc100", "cleanrl/offline/bc_discrete.py", ["--frac", "1.0"]),
    ("bc10", "cleanrl/offline/bc_discrete.py", ["--frac", "0.1"]),
    ("dt", "cleanrl/offline/dt_discrete.py", []),
    ("cql", "cleanrl/offline/cql_discrete.py", ["--cql_alpha", "1.0"]),
    ("dqn_offline", "cleanrl/offline/cql_discrete.py", ["--cql_alpha", "0.0"]),
    ("dqn_cont", "cleanrl/offline/cql_discrete.py", ["--cql_alpha", "0.0", "--init_from_checkpoint"]),
    ("dqn_cont_top10", "cleanrl/offline/cql_discrete.py",
     ["--cql_alpha", "0.0", "--init_from_checkpoint", "--top_frac", "0.1", "--top_prob", "0.5"]),
]


def get_wandb_key():
    """Read the wandb API key from ~/.netrc (never printed)."""
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    netrc = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc):
        return None
    tokens = open(netrc).read().split()
    try:
        return tokens[tokens.index("api.wandb.ai") + 4]
    except (ValueError, IndexError):
        return None


def collect_jobs(a):
    jobs = []
    for env_id in a.env_ids:
        for seed in a.seeds:
            jobs.append(["cleanrl/dqn.py", "--seed", str(seed), "--env_id", env_id,
                         "--total_timesteps", str(a.total_timesteps),
                         "--save_dataset_fractions", *[str(f) for f in a.fractions],
                         "--track", "--log_dir", "/scratch/", *a.extra])
    return jobs


def remote_datasets(a, env_id):
    """List datasets saved on the cluster's scratch, as paths inside the container."""
    c = CLUSTERS[a.cluster]
    pattern = f"{c['scratch']}/datasets/{env_id}__dqn__*/step_*.npz"
    out = subprocess.run(["ssh", c["ssh"], f"ls {pattern}"], capture_output=True, text=True)
    paths = [p for p in out.stdout.split() if p.endswith(".npz")]
    return ["/scratch" + p[len(c["scratch"]):] for p in paths]


def offline_jobs(a):
    datasets = []
    for env_id in a.env_ids:
        datasets += remote_datasets(a, env_id)
    if a.steps:
        datasets = [d for d in datasets if any(d.endswith(f"step_{s}.npz") for s in a.steps)]
    print(f"Found {len(datasets)} datasets on {a.cluster}")
    variants = [v for v in OFFLINE_VARIANTS if not a.variants or v[0] in a.variants]
    jobs = []
    for ds in datasets:
        for _, script, vargs in variants:
            for seed in a.seeds:
                jobs.append([script, "--dataset", ds, "--seed", str(seed), *vargs,
                             "--track", "--log_dir", "/scratch/", *a.extra])
    return jobs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["collect", "offline"])
    p.add_argument("--cluster", default="mila", choices=list(CLUSTERS))
    p.add_argument("--env_ids", nargs="+", default=["MinAtar/SpaceInvaders-v1"])
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="default: 1-5 for collect, 1-3 for offline")
    p.add_argument("--total_timesteps", type=int, default=10_000_000)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="offline: only datasets saved at these global steps")
    p.add_argument("--variants", nargs="+", default=None, choices=[v[0] for v in OFFLINE_VARIANTS],
                   help="offline: subset of interventions (default: all)")
    p.add_argument("--hours", type=float, default=None, help="time limit (default: 12 collect, 8 offline)")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--ram_gib", type=int, default=16)
    p.add_argument("--dry_run", action="store_true")
    a, extra = p.parse_known_args()
    a.extra = extra  # passed through to every job, e.g. --total_updates 100000
    if a.seeds is None:
        a.seeds = [1, 2, 3, 4, 5] if a.stage == "collect" else [1, 2, 3]
    if a.hours is None:
        a.hours = 12 if a.stage == "collect" else 8
    return a


ARGS = parse_args()


def main(_):
    import asyncio

    from xmanager import xm

    a = ARGS
    jobs = collect_jobs(a) if a.stage == "collect" else offline_jobs(a)
    c = CLUSTERS[a.cluster]
    print(f"{len(jobs)} jobs on {a.cluster} ({a.cpus} CPU, {a.ram_gib} GiB, {a.hours} h):")
    for j in jobs[:10]:
        print("  python " + shlex.join(j))
    if len(jobs) > 10:
        print(f"  ... and {len(jobs) - 10} more")
    if a.dry_run or not jobs:
        return

    import xm_slurm
    import xm_slurm.contrib.clusters
    from xm_slurm.contrib.clusters import drac

    wandb_key = get_wandb_key()
    if wandb_key is None:
        print("WARNING: no wandb API key found (~/.netrc or WANDB_API_KEY); runs will not sync.")
    env_vars = {"WANDB_API_KEY": wandb_key} if wandb_key else {}

    if a.cluster == "mila":
        cluster = xm_slurm.contrib.clusters.mila(user=c["user"])
    else:
        cluster = getattr(drac, a.cluster)(user=c["user"], account=c["account"])

    @xm.run_in_asyncio_loop
    async def launch():
        async with xm_slurm.create_experiment(f"sub-optimality-{a.stage}") as experiment:
            [executable] = experiment.package([
                xm_slurm.docker_container(
                    executor_spec=xm_slurm.Slurm.Spec(tag=IMAGE),
                    context=__import__("pathlib").Path(REPO_ROOT),
                    env_vars=env_vars,
                ),
            ])
            executor = xm_slurm.Slurm(
                requirements=xm_slurm.JobRequirements(CPU=a.cpus, RAM=a.ram_gib * xm.GiB, cluster=cluster),
                time=dt.timedelta(hours=a.hours),
                partition=c["partition"],
            )

            async def make_job(wu, args):
                await wu.add(xm.Job(executable=executable, executor=executor, args=args))

            wus = await experiment.add(make_job, [xm_slurm.JobArgs(args=j) for j in jobs])
            print(f"experiment_id {experiment.experiment_id}: submitted {len(wus)} work units to {a.cluster}")
            statuses = await asyncio.gather(*[wu.get_status() for wu in wus[:3]])
            for wu, s in zip(wus, statuses):
                print(f"  work unit {wu.work_unit_id}: {s}")

    launch()


if __name__ == "__main__":
    app.run(main, argv=sys.argv[:1])
