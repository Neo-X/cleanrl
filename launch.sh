#!/bin/bash

#SBATCH --partition=long-cpu                           # Ask for unkillable job
#SBATCH --cpus-per-task=16                                # Ask for 2 CPUs
#SBATCH --mem=64G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:55:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch

module load cudatoolkit/12.1 miniconda/3
conda activate cleanrl

python cleanrl/dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality
python cleanrl/dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/Asterix-v0 --track --wandb-project-name sub-optimality

python cleanrl/ppo.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality
python cleanrl/ppo.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/Asterix-v0 --track --wandb-project-name sub-optimality
