"""Online Decision Transformer: dqn.py with the Q-learner replaced by a discrete DT (cleanrl/offline/dt_discrete.py).

Same env, epsilon-greedy schedule, train frequency and gap tracking as dqn.py. Completed episodes go into a
trajectory buffer (the most recent `buffer_size` transitions); every `train_frequency` steps the DT takes one
gradient step on windows sampled from it. The DT acts greedily, conditioned on `target_return` (default: the
top-k% return of the episodes in the buffer, i.e. hindsight conditioning on the best recent experience).
"""
import os
# Limit threads for OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# Limit threads for MKL
os.environ["MKL_NUM_THREADS"] = "1"
# Limit threads for OpenMP (a common standard for parallel programming)
os.environ["OMP_NUM_THREADS"] = "1"
# Limit threads for VecLib (another potential backend)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# Limit threads for NumExpr (if used for expression evaluation)
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import collections
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
from torch.utils.tensorboard import SummaryWriter
import buffer_gap

# ===================== load the reward module ===================== #
import sys
sys.path.append("../")
from rllte.xplore.reward import RND
# ===================== load the reward module ===================== #

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import toy_landscape  # noqa: F401  registers Toy/Landscape-* envs with Gymnasium
except Exception as e:
    print(f"[toy_landscape] not registered: {e}")

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
    wandb_entity: str = "glen-berseth"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Asterix-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.10
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.25
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    intrinsic_rewards: str = False
    """Whether to use intrinsic rewards"""
    top_return_buff_percentage: float = 0.05
    """The top percent of the buffer for computing the optimality gap"""
    return_buffer_size: int = 1000
    """The size of the return buffer for computing the optimality gap"""
    log_dir: str = False
    """The directory to save the logs"""
    job_id : int = 0
    """The job id for the slurm job"""
    intrinsic_reward_scale: float = 1.0
    """The scale of the intrinsic reward"""
    old_wrappers: bool = False
    """Whether to use the old wrappers for the Atari environments"""
    num_layers: int = 1
    """The number of layers in the neural network"""
    num_units: int = 120
    """The number of units in the neural network"""
    use_layer_norm: bool = False
    """Whether to use layer normalization"""
    save_dataset_fractions: tuple[float, ...] = ()
    """fractions of total_timesteps at which to save the replay buffer + model as an offline dataset, e.g. 0.25 0.5 0.75 1.0"""
    dataset_dir: str = ""
    # Decision Transformer
    target_return: str = "topk"
    """return to condition on when acting: 'topk' (top-k% of buffer episodes), 'max' (best in buffer) or a number"""
    dt_embedding_dim: int = 128
    dt_num_layers: int = 3
    dt_num_heads: int = 1
    dt_seq_len: int = 20
    dt_max_timestep: int = 10000
    """timestep embeddings; later timesteps are clamped"""
    dt_dropout: float = 0.1
    dt_learning_rate: float = 1e-4
    dt_weight_decay: float = 1e-4
    dt_warmup_steps: int = 10000
    dt_batch_size: int = 64
    dt_reward_scale: float = 0.01
    """where to save offline datasets (default: <log_dir>datasets/<run_name>)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        import buffer_gap
        env = buffer_gap.RecordEpisodeStatisticsV2(env)
        env.action_space.seed(seed)

        return env

    return thunk


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
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    args = tyro.cli(Args)
    if str(args.intrinsic_rewards).lower() in ("", "none", "false"):
        args.intrinsic_rewards = False  # allow `--intrinsic_rewards none` from shell launchers
    # args.seed = int(os.environ.get("SLURM_PROCID", args.seed))
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.job_id}__{args.seed}__{int(time.time())}"
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
            dir=args.log_dir+f"wandb/{run_name}"
        )
    if args.log_dir:
        writer = SummaryWriter(args.log_dir+f"runs/{run_name}", max_queue=1000)    
    else:
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
    envs = buffer_gap.SyncVectorEnvV2(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ===================== build the reward ===================== #
    if args.intrinsic_rewards:
        from rllte.xplore.reward import RND, E3B
        klass = globals()[args.intrinsic_rewards]
        irs = klass(envs=envs, device=device, encoder_model="flat", obs_norm_type="none", beta=args.intrinsic_reward_scale)
    # ===================== build the reward ===================== #
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline"))
    from dt_discrete import DecisionTransformer
    n_actions = envs.single_action_space.n
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    model = DecisionTransformer(obs_dim, n_actions, args.dt_seq_len, args.dt_max_timestep, args.dt_embedding_dim,
                                args.dt_num_layers, args.dt_num_heads, args.dt_dropout, args.dt_dropout,
                                args.dt_dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.dt_learning_rate, weight_decay=args.dt_weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda n: min((n + 1) / args.dt_warmup_steps, 1))
    trajs = collections.deque()  # completed episodes: dict(obs, actions, rtg)
    n_buffer = 0

    def target_return():
        rets = sorted(t["return"] for t in trajs)
        if args.target_return == "max":
            return rets[-1]
        if args.target_return == "topk":
            k = max(int(args.top_return_buff_percentage * len(rets)), 1)
            return float(np.mean(rets[-k:]))
        return float(args.target_return)

    class DTActor:
        """Keeps the episode context (states, actions, returns-to-go) for autoregressive acting."""

        def reset(self, target):
            self.states, self.actions, self.rtgs = [], [], [target * args.dt_reward_scale]

        @torch.no_grad()
        def act(self, obs):
            self.states.append(np.asarray(obs, np.float32).reshape(-1))
            L, t = args.dt_seq_len, len(self.states)
            lo = max(0, t - L)
            S = torch.as_tensor(np.stack(self.states[lo:]), device=device).unsqueeze(0)
            A = torch.as_tensor(self.actions[lo:] + [0], device=device).long().unsqueeze(0)
            R = torch.as_tensor(self.rtgs[lo:t], device=device).float().unsqueeze(0)
            T = torch.clamp(torch.arange(lo, t, device=device), max=args.dt_max_timestep - 1).unsqueeze(0)
            model.eval()
            a = int(model(S, A, R, T)[0, -1].argmax().item())
            model.train()
            return a

        def observe(self, action, reward):
            self.actions.append(int(action))
            self.rtgs.append(self.rtgs[-1] - reward * args.dt_reward_scale)

    def sample_batch():
        lens = np.array([len(t["actions"]) for t in trajs])
        L, B = args.dt_seq_len, args.dt_batch_size
        S = np.zeros((B, L, obs_dim), np.float32); A = np.zeros((B, L), np.int64)
        R = np.zeros((B, L), np.float32); T = np.zeros((B, L), np.int64); M = np.zeros((B, L), np.float32)
        for i, ti in enumerate(np.random.choice(len(trajs), size=B, p=lens / lens.sum())):
            tr = trajs[ti]
            st = np.random.randint(0, lens[ti]); n = min(L, lens[ti] - st)
            S[i, :n], A[i, :n], R[i, :n] = tr["obs"][st:st + n], tr["actions"][st:st + n], tr["rtg"][st:st + n]
            T[i] = np.minimum(np.arange(st, st + L), args.dt_max_timestep + L - 1)
            M[i, :n] = 1
        return [torch.as_tensor(x, device=device) for x in (S, A, R, T, M)]

    actor = DTActor()
    actor.reset(0.0)
    ep_obs, ep_actions, ep_rewards = [], [], []

    #====================== optimality gap computation library ======================#
    import buffer_gap
    eval_envs = buffer_gap.SyncVectorEnvV2(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    gap_stats = buffer_gap.BufferGapV2(args.return_buffer_size, args.top_return_buff_percentage, None, device, args, eval_envs)

    def eval_deterministic_dt(best=False, _orig=gap_stats.eval_deterministic):
        """BufferGapV2.eval_deterministic, but the DT needs its context and the rewards to update returns-to-go."""
        if best or not trajs:
            return _orig(best=True) if best else None
        ev = DTActor(); ev.reset(target_return())
        obs, _ = eval_envs.reset(seed=args.seed)
        for _ in range(100000):
            a = ev.act(obs[0])
            obs, reward, _, _, infos = eval_envs.step(np.array([a]))
            ev.observe(a, reward[0])
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        return info["episode"]["return"]
        return None
    gap_stats.eval_deterministic = eval_deterministic_dt
    #====================== optimality gap computation library ======================#

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    return_ = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        greedy = actor.act(obs[0]) if trajs else None  # always run so the context stays complete
        if greedy is None or random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = np.array([greedy])

        # TRY NOT TO MODIFY: execute the game and log data.
        # print("actions", actions)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        return_ += rewards
        infos["return"] = return_

        # ===================== watch the interaction ===================== #
        if args.intrinsic_rewards:
            irs.watch(observations=obs, actions=actions, 
                    rewards=rewards, terminateds=terminations, 
                    truncateds=truncations, next_observations=next_obs)
        # ===================== watch the interaction ===================== #

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    gap_stats.add(info["episode"])
                    if global_step % args.plot_freq == 0:
                        print(f"global_step={global_step}, episodic_return={info['episode']['return']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["return"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["length"], global_step)
                        #====================== optimality gap computation logging ======================#
                        gap_stats.plot_gap(writer, global_step)
                        #====================== optimality gap computation logging ======================#

        # store the episode for the DT; extrinsic rewards only
        actor.observe(actions[0], rewards[0])
        ep_obs.append(np.asarray(obs[0], np.float32).reshape(-1)); ep_actions.append(int(actions[0]))
        ep_rewards.append(float(rewards[0]))
        if terminations[0] or truncations[0]:
            r = np.asarray(ep_rewards, np.float32)
            trajs.append(dict(obs=np.stack(ep_obs), actions=np.asarray(ep_actions, np.int64),
                              rtg=(np.cumsum(r[::-1])[::-1] * args.dt_reward_scale).astype(np.float32)))
            trajs[-1]["return"] = float(r.sum())
            n_buffer += len(r)
            while n_buffer - len(trajs[0]["actions"]) >= args.buffer_size:
                n_buffer -= len(trajs.popleft()["actions"])
            ep_obs, ep_actions, ep_rewards = [], [], []
            actor.reset(target_return())

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and trajs and global_step % args.train_frequency == 0:
            S, A, R, T, M = sample_batch()
            logits = model(S, A, R, T, padding_mask=~M.bool())
            loss = F.cross_entropy(logits.reshape(-1, n_actions), A.reshape(-1), reduction="none")
            loss = (loss * M.reshape(-1)).sum() / M.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            if global_step % args.plot_freq == 0:
                writer.add_scalar("losses/dt_loss", loss.item(), global_step)
                writer.add_scalar("charts/target_return", target_return(), global_step)
                writer.add_scalar("charts/buffer_episodes", len(trajs), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        torch.save(model.state_dict(), f"runs/{run_name}/{args.exp_name}.cleanrl_model")

    envs.close()
    writer.close()
