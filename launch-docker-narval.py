import datetime as dt

import asyncio
from absl import app
from xmanager import xm
import os

import xm_slurm
import xm_slurm.contrib.clusters

# --- CONFIGURATION ---
wandb_api_key = os.environ.get("WANDB_API_KEY")

@xm.run_in_asyncio_loop
async def main(_):
    async with xm_slurm.create_experiment("My Experiment") as experiment:
        # Step 1: Specify the executor specification
        executor_spec = xm_slurm.Slurm.Spec(tag="gberseth/cleanrl-slurm:latest")

        # Step 2: Specify the executable and package it
        [executable] = experiment.package(
            [
                xm_slurm.docker_container(
                    executor_spec=executor_spec,
                    args=["cleanrl/rainbow_atari.py", "--track", "--total_timesteps", "25000000",
                          "--wandb-project-name", "sub-optimality", "--log_dir", "/network/scratch/g/glen.berseth/"],
                    env_vars={"WANDB_API_KEY": wandb_api_key},
                ),
            ],
        )

        executor=xm_slurm.Slurm(
                    requirements=xm_slurm.JobRequirements(
                        CPU=8,
                        RAM=64 * xm.GiB,
                        GPU=1,
                        replicas=1,
                        cluster=xm_slurm.contrib.clusters.drac.narval(user="gberseth", account="rrg-gberseth"),
                    ),
                    time=dt.timedelta(hours=167),
                    requeue_on_timeout=False,
                )

        async def make_job(wu: xm.WorkUnit, args: xm.UserArgs) -> None:
            await wu.add(
                xm.Job(
                    executable=executable,
                    executor=executor,
                    args=xm.merge_args(
                        args,
                        {"log_dir": f"/scratch/{wu.experiment_id}/{wu.work_unit_id}"},
                    ),
                )
            )

        args = [xm_slurm.JobArgs(args={"seed": scale}) for scale in range(4)]
        wus = await experiment.add(make_job, args)

        for wu, status in zip(wus, await asyncio.gather(*[wu.get_status() for wu in wus])):
            print(f"Work Unit {wu.work_unit_id} Status: {status}")

        print("experiment_id", experiment.experiment_id)
        # await asyncio.gather(*[wu.wait_until_complete() for wu in wus])
        # print("All jobs finished!")


if __name__ == "__main__":
    app.run(main)
