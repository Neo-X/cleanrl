"""Discrete-action CQL (and plain offline DQN) on a saved online replay buffer.

CORL's algorithms/offline/cql.py (Apache-2.0) is continuous-only (tanh-Gaussian actor,
sampled-action logsumexp). For discrete actions we follow Kumar et al. 2020, "Conservative
Q-Learning for Offline RL", Sec. 3 CQL(H) with exact logsumexp over actions, on top of the
same DQN update as cleanrl/dqn.py (MSE TD loss, hard target updates, same network):

    L = TD(Q) + alpha * E_{s,a~D}[ logsumexp_a' Q(s,a') - Q(s,a) ]

Variants, all trained only on the fixed buffer D (no new experience):
    --cql_alpha 1.0                       offline RL (CQL) from scratch
    --cql_alpha 0.0                       offline DQN from scratch
    --init_from_checkpoint --cql_alpha 0  continue the online agent with standard replay
    ... --top_frac 0.1 --top_prob 0.5     same, oversampling the top-10% trajectories

    python cleanrl/offline/cql_discrete.py --dataset <dir>/step_5000000.npz
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
    """path to a step_<N>.npz saved by cleanrl/dqn.py --save_dataset_fractions"""
    cql_alpha: float = 1.0
    """weight of the CQL(H) regularizer; 0 = offline DQN"""
    init_from_checkpoint: bool = False
    """start from the online agent's Q-network saved with the dataset"""
    top_frac: float = 0.0
    """oversample transitions from the top `top_frac` complete trajectories (0 = off)"""
    top_prob: float = 0.0
    """probability that a batch element is drawn from those top trajectories"""
    seed: int = 1
    total_updates: int = 200_000
    batch_size: int = 128
    """defaults below match cleanrl/dqn.py"""
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    target_update_freq: int = 50
    """in gradient steps; dqn.py updates the target every 500 env steps = 50 updates"""
    num_layers: int = 1
    num_units: int = 120
    use_layer_norm: bool = False
    eval_freq: int = 10_000
    eval_episodes: int = 10
    eval_seed: int = 10_000
    eval_max_steps: int = 10_000
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
    name = "cql" if args.cql_alpha > 0 else "dqn_offline"
    if args.init_from_checkpoint:
        name += "_cont"
    if args.top_prob > 0:
        name += f"_top{int(round(args.top_frac * 100))}"
    logger = common.Logger(args, name, args.dataset)
    refs = common.compute_refs(data, args, env_id, device)
    sampler = common.TransitionSampler(data, device, args.top_frac, args.top_prob)

    if args.init_from_checkpoint:
        net_kwargs = common.online_net_kwargs(data)
    else:
        net_kwargs = dict(num_layers=args.num_layers, num_units=args.num_units, use_layer_norm=args.use_layer_norm)
    q = common.QNetwork(env.observation_space.shape, env.action_space.n, **net_kwargs).to(device)
    if args.init_from_checkpoint:
        q.load_state_dict(data["online_state_dict"])
    target = common.QNetwork(env.observation_space.shape, env.action_space.n, **net_kwargs).to(device)
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
