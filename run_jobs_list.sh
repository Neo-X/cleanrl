#/bin/bash
## Code to run all the jobs for this codebase 

## Discrete RL Envs
strings=(
    "MinAtar/SpaceInvaders-v0"
    "MinAtar/Asterix-v0"
    # "MinAtar/Breakout-v0"
    # "MinAtar/Seaquest-v0"
    # "MinAtar/Freeway-v0"
    "LunarLander-v2"
)
for env in "${strings[@]}"; do
#     # echo "$env"
### Standard experiments
    sbatch --array=1-5 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID=$env,ARGSS='--track --total_timesteps 25000000 --intrinsic_reward_scale=0.2' launch.sh
    sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID=$env,ARGSS='--track --total_timesteps 25000000 --num_envs 4 --intrinsic_reward_scale=0.2' launch.sh
    sbatch --array=1-5 --export=ALL,ALG='cleanrl/pqn.py',ENV_ID=$env,ARGSS='--track --total_timesteps 25000000 --num_envs 4 --intrinsic_reward_scale=0.2' launch.sh
### with intrinsic rewards
    sbatch --array=1-5 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID=$env,ARGSS='--track --total_timesteps 25000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launch.sh
    sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID=$env,ARGSS='--track --total_timesteps 25000000 --num_envs 4 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launch.sh
    sbatch --array=1-5 --export=ALL,ALG='cleanrl/pqn.py',ENV_ID=$env,ARGSS='--track --total_timesteps 25000000 --num_envs 4 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launch.sh

### Network scaling experiments with 16 or more layers
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID=$env,ARGSS='--num_layers=16 --total_timesteps 10000000 ' launch.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID=$env,ARGSS='--num_layers=16 --total_timesteps 10000000 --num_envs 4' launch.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID=$env,ARGSS='--num_layers=16 --use_layer_norm --total_timesteps 10000000 ' launch.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID=$env,ARGSS='--num_layers=16 --use_layer_norm --total_timesteps 10000000 --num_envs 4' launch.sh
done

##Continuous RL envs
strings=(
    # "Walker2d-v4"
    # "HalfCheetah-v4"
    # "Humanoid-v4"
    # "BipedalWalker-v3"
)
for env in "${strings[@]}"; do
#     # echo "$env"
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--total_timesteps 10000000' launch.sh
#     sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--num_layers=16 --num_envs 4 --total_timesteps 5000000' launch.sh
#     sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--num_layers=32 --num_envs 4 --total_timesteps 5000000' launch.sh
#     sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--num_layers=64 --num_envs 4 --total_timesteps 5000000' launch.sh
#     sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--num_layers=128 --num_envs 4 --total_timesteps 5000000' launch.sh
#     sbatch --array=1-5 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--num_layers=256 --num_envs 4 --total_timesteps 5000000' launch.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--num_layers=16 --use_layer_norm --num_envs 4 --total_timesteps 5000000' launch.sh
    sbatch --array=1-10 --export=ALL,ALG='cleanrl/sac_continuous_action.py',ENV_ID=$env,ARGSS='--total_timesteps 10000000' launch.sh ## SAC Experiments Normal
done

## Atari RL envs
strings=(
    # "MontezumaRevengeNoFrameskip-v4"
    # "AsterixNoFrameskip-v4"
    # "SpaceInvadersNoFrameskip-v4"
    # "PitfallNoFrameskip-v4"
    # "BattleZoneNoFrameskip-v4"
    # "NameThisGameNoFrameskip-v4"
    # "PhoenixNoFrameskip-v4"
)
for env in "${strings[@]}"; do
    echo "$env"
    ## PPO Experiments
    sbatch --array=5-10 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 50000000' launchGPU.sh ## Normal PPO
    sbatch --array=5-10 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 50000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launchGPU.sh ## PPO with RND
    # sbatch --array=5-10 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID=$env,ARGSS='--network_type ResNet --total_timesteps 50000000' --time=6-00:00:00 launchGPU.sh ## PPO with ResNet
    ## DQN Experiments
    sbatch --array=5-10 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 50000000' launchGPU.sh ## Normal
    sbatch --array=5-10 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 50000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launchGPU.sh ## with RND
    # sbatch --array=5-10 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID=$env,ARGSS='--network_type ResNet --total_timesteps 50000000' --time=6-00:00:00 launchGPU.sh ## with ResNet
    ## PQN Experiments
    sbatch --array=1-10 --export=ALL,ALG='cleanrl/pqn_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 50000000' launchGPU.sh ## Normal
    sbatch --array=1-10 --export=ALL,ALG='cleanrl/pqn_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 50000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launchGPU.sh ## with RND
    # sbatch --array=1-10 --export=ALL,ALG='cleanrl/pqn_atari.py',ENV_ID=$env,ARGSS='--network_type ResNet --total_timesteps 50000000' --time=6-00:00:00 launchGPU.sh ## with ResNet
done