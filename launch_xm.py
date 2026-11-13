"""Launch the fixed-data intervention experiments with xm-slurm on Mila or DRAC (Narval, Rorqual).

The container is built from this repo's Dockerfile, so the local working tree (including
uncommitted changes) is what runs, and pushed to Docker Hub (`IMAGE`). Every cluster mounts
its scratch at /scratch inside the container, so all logs and datasets go to /scratch/:

    Mila:  /network/scratch/g/glen.berseth/   DRAC: /scratch/gberseth/

Run with the xm-slurm environment:

    # 1. online DQN, saving (model, replay buffer) at 25/50/75/100% of training
    ../xm-slurm/.venv/bin/python launch_xm.py collect --cluster narval --env_ids MinAtar/SpaceInvaders-v1

    # online CQL (dqn.py + CQL(H) regulariser), everything else identical to DQN
    ../xm-slurm/.venv/bin/python launch_xm.py online --cluster mila --algo dqn_cql --env_ids MinAtar/SpaceInvaders-v0 --total_timesteps 25000000

    # same, with RND exploration (datasets go to datasets/<env>__dqn_rnd__*)
    ../xm-slurm/.venv/bin/python launch_xm.py collect --cluster mila --rnd_scale 0.2

    # 2. every offline intervention on every dataset found on that cluster's scratch
    ../xm-slurm/.venv/bin/python launch_xm.py offline --cluster narval --env_ids MinAtar/SpaceInvaders-v1
    ../xm-slurm/.venv/bin/python launch_xm.py offline --cluster mila --exp_name dqn_rnd --steps 10000000
    ../xm-slurm/.venv/bin/python launch_xm.py offline --cluster mila --env_ids MinAtar/SpaceInvaders-v0 --collect_id 38

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
    "mila": dict(ssh="mila", user="glen.berseth", account=None, partition="long-cpu", gpu_partition="long",
                 scratch="/network/scratch/g/glen.berseth"),
    "narval": dict(ssh="narval", user="gberseth", account="rrg-gberseth", partition=None, gpu_partition=None,
                   scratch="/scratch/gberseth"),
    "rorqual": dict(ssh="rorqual", user="gberseth", account="rrg-gberseth", partition=None, gpu_partition=None,
                    scratch="/scratch/gberseth"),
}

# Resource profiles. "gpu" mirrors launchGPU.sh (long partition, 1 GPU, 8 CPUs, 64G, 11:59, --no-requeue);
# "cpu" mirrors launch.sh's long-cpu jobs. Jobs are never requeued: the scripts don't resume from checkpoints.
PROFILES = {
    "gpu": dict(gpus=1, cpus=8, ram_gib=64, hours=11 + 59 / 60),
    "cpu": dict(gpus=0, cpus=4, ram_gib=16, hours=8),
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
    """One xm-slurm job array: fixed args shared by every trial, plus the swept (env_id, seed) trials."""
    rnd = ["--intrinsic_rewards", "RND", "--intrinsic_reward_scale", str(a.rnd_scale)] if a.rnd_scale > 0 else []
    fixed = [f"cleanrl/{a.algo}.py", "--exp_name", a.exp_name, "--total_timesteps", str(a.total_timesteps), *rnd,
             *(["--save_dataset_fractions", *[str(f) for f in a.fractions]] if a.stage == "collect" else []),
             "--track", "--log_dir", "/scratch/", *a.cuda, *a.extra]
    sweep = [{"env_id": env_id, "seed": seed} for env_id in a.env_ids for seed in a.seeds]
    return [(fixed, sweep)]


def remote_datasets(a, env_id):
    """List datasets saved on the cluster's scratch, as paths inside the container."""
    c = CLUSTERS[a.cluster]
    run = f"{a.collect_id}__*" if a.collect_id else "*"
    pattern = f"{c['scratch']}/datasets/{env_id}__{a.exp_name}__{run}/step_*.npz"
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
    sweep = [{"dataset": ds, "seed": seed} for ds in datasets for seed in a.seeds]
    # one job array per intervention (each runs a different script with different fixed args)
    return [([script, *vargs, "--track", "--log_dir", "/scratch/", *a.cuda, *a.extra], sweep)
            for _, script, vargs in variants]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["online", "collect", "offline"],
                   help="online: train an online agent; collect: same, also saving datasets; offline: train on saved datasets")
    p.add_argument("--cluster", default="mila", choices=list(CLUSTERS))
    p.add_argument("--env_ids", nargs="+", default=["MinAtar/SpaceInvaders-v1"])
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="default: 1-5 for collect, 1-3 for offline")
    p.add_argument("--total_timesteps", type=int, default=10_000_000)
    p.add_argument("--algo", default="dqn", choices=["dqn", "dqn_cql", "dqn_atari", "dqn_atari_cql"],
                   help="online/collect: learner (dqn_cql = dqn.py + CQL(H) term, see --cql_alpha passthrough)")
    p.add_argument("--rnd_scale", type=float, default=0.0,
                   help="collect: train DQN with RND at this intrinsic reward scale (0 = off; paper uses 0.2)")
    p.add_argument("--exp_name", default=None,
                   help="dataset/run name: collect writes datasets/<env>__<exp_name>__*, offline reads them "
                        "(default: dqn, or dqn_rnd when --rnd_scale > 0)")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--collect_id", default=None,
                   help="offline: only datasets from this collect experiment/job id (datasets/<env>__<exp_name>__<id>__*)")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="offline: only datasets saved at these global steps")
    p.add_argument("--variants", nargs="+", default=None, choices=[v[0] for v in OFFLINE_VARIANTS],
                   help="offline: subset of interventions (default: all)")
    p.add_argument("--profile", choices=list(PROFILES), default=None,
                   help="resources: gpu = launchGPU.sh (default for collect on non-MinAtar envs, e.g. Atari), "
                        "cpu = launch.sh (default for MinAtar and for offline)")
    p.add_argument("--hours", type=float, default=None, help="override the profile's time limit")
    p.add_argument("--cpus", type=int, default=None, help="override the profile's CPUs")
    p.add_argument("--ram_gib", type=int, default=None, help="override the profile's RAM")
    p.add_argument("--dry_run", action="store_true")
    a, extra = p.parse_known_args()
    a.extra = extra  # passed through to every job, e.g. --total_updates 100000
    if a.exp_name is None:
        a.exp_name = a.algo + ("_rnd" if a.rnd_scale > 0 else "")
    if a.seeds is None:
        a.seeds = [1, 2, 3] if a.stage == "offline" else [1, 2, 3, 4, 5]
    if a.profile is None:
        minatar_only = all(e.startswith("MinAtar/") for e in a.env_ids)
        a.profile = "gpu" if a.stage != "offline" and not minatar_only else "cpu"
    prof = dict(PROFILES[a.profile])
    if a.stage != "offline" and a.profile == "cpu":
        prof["hours"] = 11 + 55 / 60  # launch.sh's limit; the non-RND MinAtar runs took ~1.5 h
    a.gpus = prof["gpus"]
    a.cpus = a.cpus or prof["cpus"]
    a.ram_gib = a.ram_gib or prof["ram_gib"]
    a.hours = a.hours or prof["hours"]
    a.cuda = ["--cuda"] if a.gpus else []
    return a


ARGS = parse_args()


def main(_):
    import asyncio

    from xmanager import xm

    a = ARGS
    arrays = offline_jobs(a) if a.stage == "offline" else collect_jobs(a)
    arrays = [(fixed, sweep) for fixed, sweep in arrays if sweep]
    c = CLUSTERS[a.cluster]
    n = sum(len(sweep) for _, sweep in arrays)
    print(f"{n} jobs in {len(arrays)} job array(s) on {a.cluster} "
          f"({a.gpus} GPU, {a.cpus} CPU, {a.ram_gib} GiB, {a.hours:.2f} h):")
    for fixed, sweep in arrays:
        print(f"  python {shlex.join(fixed)}  x {len(sweep)} trials, e.g. {sweep[0]}")
    if a.dry_run or not arrays:
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
            req = dict(CPU=a.cpus, RAM=a.ram_gib * xm.GiB, cluster=cluster)
            if a.gpus:
                req["GPU"] = a.gpus
            executor = xm_slurm.Slurm(
                requirements=xm_slurm.JobRequirements(**req),
                time=dt.timedelta(hours=a.hours),
                partition=c["gpu_partition"] if a.gpus else c["partition"],
                requeue=False,
                requeue_on_timeout=False,
            )

            def make_job(fixed):
                # like launchGPU.sh's --job_id $SLURM_JOB_ID: tag runs/datasets with the experiment id
                if fixed[0].startswith("cleanrl/dqn"):
                    fixed = [*fixed, "--job_id", str(experiment.experiment_id)]

                async def job(wu, args):
                    await wu.add(xm.Job(executable=executable, executor=executor, args=xm.merge_args(fixed, args)))
                return job

            wus = []
            for fixed, sweep in arrays:
                wus += await experiment.add(make_job(fixed), [xm_slurm.JobArgs(args=t) for t in sweep])
            print(f"experiment_id {experiment.experiment_id}: submitted {len(wus)} work units to {a.cluster}")
            statuses = await asyncio.gather(*[wu.get_status() for wu in wus[:3]])
            for wu, s in zip(wus, statuses):
                print(f"  work unit {wu.work_unit_id}: {s}")

    launch()


if __name__ == "__main__":
    app.run(main, argv=sys.argv[:1])
