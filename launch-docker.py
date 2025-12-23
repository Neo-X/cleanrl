import datetime as dt

from absl import app
from xmanager import xm

import xm_slurm
import xm_slurm.contrib.clusters


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
                    args=["cleanrl/ppo_continuous_action.py", "--track"],
                    env_vars={"WANDB_API_KEY": "10000"},
                ),
            ],
        )

        wu = await experiment.add(
            xm.Job(
                executable=executable,
                executor=xm_slurm.Slurm(
                    requirements=xm_slurm.JobRequirements(
                        CPU=1,
                        RAM=16 * xm.GiB,
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
