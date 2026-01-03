import datetime as dt

from absl import app
from xmanager import xm

import xm_slurm
import xm_slurm.contrib.clusters

import os
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
                    args=["cleanrl/rainbow_atari.py", "--track"],
                    env_vars={"WANDB_API_KEY": wandb_api_key},
                ),
            ],
        )

        wu = await experiment.add(
            xm.Job(
                executable=executable,
                executor=xm_slurm.Slurm(
                    requirements=xm_slurm.JobRequirements(
                        CPU=16,
                        RAM=64 * xm.GiB,
                        GPU=1,
                        cluster=xm_slurm.contrib.clusters.mila(user="glen.berseth"),
                    ),
                    time=dt.timedelta(hours=1),
                ),
            )
        )

        print("experiment_id", experiment.experiment_id)
        await wu.wait_until_complete()
        print(f"Job finished executing with status {await wu.get_status()}")


if __name__ == "__main__":
    app.run(main)
