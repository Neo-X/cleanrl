# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
# Limit threads for OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
# Limit threads for MKL
os.environ["MKL_NUM_THREADS"] = "4"
# Limit threads for OpenMP (a common standard for parallel programming)
os.environ["OMP_NUM_THREADS"] = "4"
# Limit threads for VecLib (another potential backend)
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# Limit threads for NumExpr (if used for expression evaluation)
os.environ["NUMEXPR_NUM_THREADS"] = "4"
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
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


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
    plot_freq: int = 1000
    """The frequency of plotting"""
    wandb_project_name: str = "sub-optimality"
    """the wandb's project name"""
    wandb_entity: str = "real-lab"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Asterix-v0"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89 ## https://github.com/jakegrigsby/super_sac/blob/92398326f04b22ea2a7d3abe2e6a626620dcead5/experiments/minatar/basic_online.gin#L47
    """coefficient for scaling the autotune entropy target"""

    intrinsic_rewards: str = False
    """Whether to use intrinsic rewards"""
    top_return_buff_percentage: float = 0.05
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
    num_units: int = 256
    """The number of units in the neural network"""
    use_layer_norm: bool = False
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
        return env

    return thunk


# def layer_init(layer, bias_const=0.0):
#     nn.init.kaiming_normal_(layer.weight)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        layers = [
            nn.Flatten(),
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.num_units)),
            nn.ReLU()]
        
        for i in range(args.num_layers):
            layers.append(layer_init(nn.Linear(args.num_units, args.num_units)))
            layers.append(nn.ReLU())
            if args.use_layer_norm:
                layers.append(nn.LayerNorm(args.num_units))

        self.conv = nn.Sequential(*layers)

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 64))
        self.fc_q = layer_init(nn.Linear(64, envs.single_action_space.n))

    def forward(self, x):
        x = self.conv(x * 1.0)
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        layers = [
            nn.Flatten(),
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.num_units)),
            nn.ReLU()]
        
        for i in range(args.num_layers):
            layers.append(layer_init(nn.Linear(args.num_units, args.num_units)))
            layers.append(nn.ReLU())
            if args.use_layer_norm:
                layers.append(nn.LayerNorm(args.num_units))

        self.conv = nn.Sequential(*layers)

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 64))
        self.fc_logits = layer_init(nn.Linear(64, envs.single_action_space.n))

    def forward(self, x):
        x = self.conv(x * 1.0)
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs
    
    def get_action_deterministic(self, x):
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=1)
        return actions


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
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
            dir="/network/scratch/g/glen.berseth/"
        )
    writer = SummaryWriter(f"runs/{run_name}", max_queue=1000)
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
    import buffer_gap
    envs = buffer_gap.SyncVectorEnvV2([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ===================== build the reward ===================== #
    if args.intrinsic_rewards:
        from rllte.xplore.reward import RND, E3B
        klass = globals()[args.intrinsic_rewards]
        irs = klass(envs=envs, device=device, encoder_model="flat", obs_norm_type="none", beta=args.intrinsic_reward_scale)
    # ===================== build the reward ===================== #

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    #====================== optimality gap computation library ======================#
    import buffer_gap
    eval_envs = buffer_gap.SyncVectorEnvV2(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    gap_stats = buffer_gap.BufferGapV2(args.return_buffer_size, args.top_return_buff_percentage, actor, device, args, eval_envs)
    #====================== optimality gap computation library ======================#

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    last_log_step = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # ===================== watch the interaction ===================== #
        if args.intrinsic_rewards:
            irs.watch(observations=obs, actions=actions, 
                    rewards=rewards, terminateds=terminations, 
                    truncateds=truncations, next_observations=next_obs)
        # ===================== watch the interaction ===================== #

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                gap_stats.add(info["episode"])
                if (global_step - last_log_step) > args.plot_freq*5:
                    last_log_step = global_step
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    #====================== optimality gap computation logging ======================#
                    gap_stats.plot_gap(writer, global_step)
                    #====================== optimality gap computation logging ======================#
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # ===================== compute the intrinsic rewards ===================== #
            # get real next observations
            
                
                
            # ===================== compute the intrinsic rewards ===================== #
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)            
                rewards_ = data.rewards
                # ===================== compute the intrinsic rewards ===================== #
                # get real next observations
                if args.intrinsic_rewards:
                    
                    intrinsic_rewards = irs.compute(samples=dict(observations=data.observations*1.0, actions=data.actions, 
                                                                rewards=data.rewards, terminateds=data.dones,
                                                                truncateds=data.dones, next_observations=data.next_observations*1.0
                                                                ))
                    rewards_ += intrinsic_rewards
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = rewards_.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                if global_step % args.plot_freq == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                    data_ = rb.sample(10000)
                    if global_step % 1000 == 0 and data_.rewards.shape[0] >= 10000:
                        writer.add_scalar("charts/rewards mean", data_.rewards.mean(), global_step)
                        writer.add_scalar("charts/rewards top 95%", torch.mean(torch.topk(data_.rewards.flatten(), 500)[0]), global_step)
                        # writer.add_scalar("charts/returns top 95%", torch.mean(torch.topk(data_.returns.flatten(), 500)[0]), global_step)
                    if args.intrinsic_rewards:
                        ## Here we iterate over the irs.metrics disctionary
                        for key, value in irs.metrics.items():
                            writer.add_scalar(key, np.mean([val[1] for val in value]), global_step)
                            irs.metrics[key] = []

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    envs.close()
    writer.close()
