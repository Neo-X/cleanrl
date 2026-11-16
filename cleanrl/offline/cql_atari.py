"""Discrete-action CQL(H) on a saved Atari replay buffer (cleanrl/dqn_atari.py --save_dataset_fractions).

Same algorithm as cql_discrete.py (Kumar et al. 2020, CQL(H) with exact logsumexp on top of the DQN update),
with the online agent's Atari settings from cleanrl/dqn_atari.py: Nature-CNN Q-network (inputs / 255),
lr 1e-4, batch 32, gamma 0.99, target network every 1000 env steps = 250 updates (train_frequency 4).
Observations are the 4-frame stacks rebuilt from the dataset's frames (common.frame_stack_index), and
evaluation uses dqn_atari.py's default wrappers (deterministic ALE, returns in clipped reward units).

    python cleanrl/offline/cql_atari.py --dataset <dir>/step_12500000.npz --cuda
"""
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import tyro

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


@dataclass
class Args:
    dataset: str
    """path to a step_<N>.npz saved by cleanrl/dqn_atari.py --save_dataset_fractions"""
    cql_alpha: float = 1.0
    """weight of the CQL(H) regularizer; 0 = offline DQN"""
    init_from_checkpoint: bool = False
    """start from the online agent's Q-network saved with the dataset"""
    top_frac: float = 0.0
    """oversample transitions from the top `top_frac` complete trajectories (0 = off)"""
    top_prob: float = 0.0
    """probability that a batch element is drawn from those top trajectories"""
    seed: int = 1
    total_updates: int = 1_000_000
    batch_size: int = 32
    """defaults below match cleanrl/dqn_atari.py"""
    learning_rate: float = 1e-4
    gamma: float = 0.99
    target_update_freq: int = 250
    """in gradient steps; dqn_atari.py updates the target every 1000 env steps = 250 updates"""
    eval_freq: int = 10_000
    eval_episodes: int = 10
    eval_seed: int = 10_000
    eval_max_steps: int = 1000
    top_k: float = 0.05
    """k for the V^{pi*_D} reference (top-k% of complete episodes in D)"""
    env_id: str = ""
    cuda: bool = False
    track: bool = False
    wandb_project_name: str = "sub-optimality"
    wandb_entity: str = "glen-berseth"
    log_dir: str = ""


def main(args: Args):
    common.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    data = common.load_dataset(args.dataset)
    env_id = common.env_id_of(data, args.env_id)
    env = common.make_env(env_id)
    assert common.is_atari(data), "cql_atari.py needs an Atari dataset (frames); use cql_discrete.py for MinAtar"
    name = "cql_atari" if args.cql_alpha > 0 else "dqn_atari_offline"
    if args.init_from_checkpoint:
        name += "_cont"
    if args.top_prob > 0:
        name += f"_top{int(round(args.top_frac * 100))}"
    logger = common.Logger(args, name, args.dataset)
    refs = common.compute_refs(data, args, env_id, device)
    sampler = common.TransitionSampler(data, device, args.top_frac, args.top_prob)

    q = common.AtariQNetwork(env.observation_space.shape, env.action_space.n).to(device)
    if args.init_from_checkpoint:
        q.load_state_dict(data["online_state_dict"])
    target = common.AtariQNetwork(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=args.learning_rate)

    if args.init_from_checkpoint:  # step 0 = the online agent itself
        logger.log_eval(0, common.evaluate(common.q_act_fn(q, device), env_id, args.eval_episodes,
                                           args.eval_seed, args.eval_max_steps), refs)

    for step in range(1, args.total_updates + 1):
        s, a, r, s2, d = sampler.sample(args.batch_size)
        with torch.no_grad():
            td_target = r + args.gamma * target(s2).max(1).values * (1 - d)
        q_all = q(s)
        q_a = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(q_a, td_target)
        cql_loss = (torch.logsumexp(q_all, dim=1) - q_a).mean()
        loss = td_loss + args.cql_alpha * cql_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % args.target_update_freq == 0:
            target.load_state_dict(q.state_dict())
        if step % 1000 == 0:
            logger.log({"losses/td_loss": td_loss.item(), "losses/cql_loss": cql_loss.item(),
                        "losses/q_values": q_a.mean().item()}, step)
        if step % args.eval_freq == 0 or step == args.total_updates:
            rets = common.evaluate(common.q_act_fn(q, device), env_id, args.eval_episodes,
                                   args.eval_seed, args.eval_max_steps)
            logger.log_eval(step, rets, refs)
    logger.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
