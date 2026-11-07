"""Discrete-action (filtered) behaviour cloning on a saved online replay buffer.

Adapted from CORL's algorithms/offline/any_percent_bc.py (Apache-2.0) to discrete actions:
cross-entropy over action logits instead of MSE on a tanh actor, undiscounted returns for
ranking trajectories (to match the paper's returns), and only *complete* episodes are
ranked. `--frac 1.0` is plain BC on all of D; `--frac 0.1` is BC-10%, i.e. imitating
the top 10% of trajectories the online agent generated.

    python cleanrl/offline/bc_discrete.py --dataset <dir>/step_5000000.npz --frac 0.1
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import tyro

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


@dataclass
class Args:
    dataset: str
    """path to a step_<N>.npz saved by cleanrl/dqn.py --save_dataset_fractions"""
    frac: float = 0.1
    """top fraction of complete trajectories (by return) to imitate; 1.0 = all transitions"""
    seed: int = 1
    total_updates: int = 200_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    num_layers: int = 1
    """network size; defaults match cleanrl/dqn.py"""
    num_units: int = 120
    use_layer_norm: bool = False
    eval_freq: int = 10_000
    eval_episodes: int = 10
    eval_seed: int = 10_000
    eval_max_steps: int = 10_000
    top_k: float = 0.05
    """k for the V^{pi*_D} reference (top-k% of complete episodes in D)"""
    env_id: str = ""
    """override the env stored with the dataset"""
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
    logger = common.Logger(args, f"bc{int(round(args.frac * 100))}", args.dataset)
    refs = common.compute_refs(data, args, env_id, device)

    if args.frac < 1.0:
        idx = common.top_trajectory_indices(data, args.frac)
    else:
        idx = np.arange(len(data["rewards"]))
    obs = torch.as_tensor(data["observations"][idx]).to(device)
    actions = torch.as_tensor(data["actions"][idx]).long().to(device)
    print(f"[bc] training on {len(idx)} / {len(data['rewards'])} transitions")

    policy = common.QNetwork(env.observation_space.shape, env.action_space.n, args.num_layers,
                             args.num_units, args.use_layer_norm).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    for step in range(1, args.total_updates + 1):
        b = torch.randint(0, len(idx), (args.batch_size,), device=device)
        logits = policy(obs[b].float())
        loss = F.cross_entropy(logits, actions[b])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            acc = (logits.argmax(1) == actions[b]).float().mean().item()
            logger.log({"losses/bc_loss": loss.item(), "losses/accuracy": acc}, step)
        if step % args.eval_freq == 0 or step == args.total_updates:
            policy.eval()
            rets = common.evaluate(common.q_act_fn(policy, device), env_id, args.eval_episodes,
                                   args.eval_seed, args.eval_max_steps)
            policy.train()
            logger.log_eval(step, rets, refs)
    logger.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
