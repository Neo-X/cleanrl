# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import buffer_gap

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Breakout-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run for each environment per update"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    anneal_lr: bool = True
    """Toggle learning rate annealing"""
    gamma: float = 0.99
    """the discount factor gamma"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    q_lambda: float = 0.65
    """the lambda for Q(lambda)"""
    """the number of iterations (computed in runtime)"""
    intrinsic_rewards: str = "RND"
    """Whether to use intrinsic rewards"""
    top_return_buff_percentage: int = 0.05
    """The top percent of the buffer for computing the optimality gap"""
    return_buffer_size: int = 1000
    """the replay memory buffer size"""
    log_dir: str = False
    """The directory to save the logs"""
    job_id : int = 0
    """The job id for the slurm job"""
    intrinsic_reward_scale: float = 1.0
    """The scale of the intrinsic reward"""
    num_layers: int = 1
    """The number of layers in the neural network"""
    num_units: int = 128
    """The number of units in the neural network"""
    use_layer_norm: bool = True
    """Whether to use layer normalization"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        import buffer_gap
        env = buffer_gap.RecordEpisodeStatisticsV2(env)
        env.action_space.seed(seed)

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        layers = [
                nn.Flatten(),
                  nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.num_units),
                  nn.ReLU()]
        for i in range(args.num_layers-1):
            layers.append(nn.Linear(args.num_units, args.num_units))
            layers.append(nn.ReLU())
            if args.use_layer_norm:
                layers.append(nn.LayerNorm(args.num_units))

        layers.extend([nn.Linear(args.num_units, 84),
                        nn.ReLU(),
                        nn.Linear(84, env.single_action_space.n),])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def get_action_deterministic(self, x):
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=1)
        return actions


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # ===================== build the reward ===================== #
    if args.intrinsic_rewards:
        from rllte.xplore.reward import RND, E3B
        klass = globals()[args.intrinsic_rewards]
        irs = klass(envs=envs, device=device, encoder_model="flat", obs_norm_type="none", beta=args.intrinsic_reward_scale)
    # ===================== build the reward ===================== #

    # agent setup
    q_network = QNetwork(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    #====================== optimality gap computation library ======================#
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    gap_stats = buffer_gap.BufferGapV2(args.return_buffer_size, args.top_return_buff_percentage, q_network, device, args, eval_envs)
    #====================== optimality gap computation library ======================#

    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # ===================== watch the interaction ===================== #
            if args.intrinsic_rewards:
                irs.watch(observations=obs[step], actions=actions[step], 
                      rewards=rewards[step], terminateds=dones[step], 
                      truncateds=dones[step], next_observations=next_obs)
            # ===================== watch the interaction ===================== #

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        #====================== optimality gap computation logging ======================#
                        gap_stats.add(info["episode"])
                        gap_stats.plot_gap(writer, global_step)
                        #====================== optimality gap computation logging ======================#

        # ===================== compute the intrinsic rewards ===================== #
        # get real next observations
        if args.intrinsic_rewards:
            real_next_obs = obs.clone()
            real_next_obs[:-1] = obs[1:]
            real_next_obs[-1] = next_obs

            intrinsic_rewards = irs.compute(samples=dict(observations=obs, actions=actions, 
                                                        rewards=rewards, terminateds=dones,
                                                        truncateds=dones, next_observations=real_next_obs
                                                        ))
            rewards += intrinsic_rewards
        # ===================== compute the intrinsic rewards ===================== #
        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        #====================== log reward statistics ===================== #
        writer.add_scalar("charts/reward mean", rewards.mean(), global_step)
        writer.add_scalar("charts/reward top 95%", torch.mean(torch.topk(rewards.flatten(), 500)[0]), global_step)
        writer.add_scalar("charts/return mean", rewards.mean(dim=0).mean(), global_step)
        writer.add_scalar("charts/avg_reward_traj top 95%", torch.mean(torch.topk(rewards.mean(dim=0).flatten(), 2)[0]), global_step)
        if args.intrinsic_rewards:
            ## Here we iterate over the irs.metrics disctionary
            for key, value in irs.metrics.items():
                writer.add_scalar(key, np.mean([val[1] for val in value]), global_step)
                irs.metrics[key] = []

    envs.close()
    writer.close()
