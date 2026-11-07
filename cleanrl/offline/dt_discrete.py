"""Discrete-action Decision Transformer (return-conditioned BC) on a saved online replay buffer.

Adapted from CORL's algorithms/offline/dt.py (Apache-2.0). Changes for discrete actions:
actions are embedded with nn.Embedding and predicted as logits (cross-entropy) instead of
a tanh head with MSE; observations (binary MinAtar grids) are flattened and not normalised;
timesteps beyond `episode_len` are clamped; only complete episodes are used (so the
returns-to-go are exact). At evaluation the model is conditioned on returns taken from D,
by default the top-k% return V^{pi*_D}(s0) and the best return in D.

    python cleanrl/offline/dt_discrete.py --dataset <dir>/step_5000000.npz
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


@dataclass
class Args:
    dataset: str
    """path to a step_<N>.npz saved by cleanrl/dqn.py --save_dataset_fractions"""
    target_returns: tuple[str, ...] = ("topk_D", "max_D")
    """returns to condition on at evaluation: 'topk_D', 'max_D', 'mean_D' or a number"""
    seed: int = 1
    # model (CORL defaults)
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 0
    """max timestep embedding; 0 = longest complete episode in D"""
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    # optimisation (CORL defaults)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    total_updates: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.01
    """CORL uses 0.001 for MuJoCo returns ~1e4; MinAtar returns are ~1e1-1e2"""
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


# ----------------------------------------------------------------------------- model (from CORL)
class TransformerBlock(nn.Module):
    def __init__(self, seq_len, embedding_dim, num_heads, attention_dropout, residual_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer("causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool))

    def forward(self, x, padding_mask=None):
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        norm_x = self.norm1(x)
        attention_out = self.attention(query=norm_x, key=norm_x, value=norm_x, attn_mask=causal_mask,
                                       key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, n_actions, seq_len=20, episode_len=1000, embedding_dim=128, num_layers=3,
                 num_heads=1, attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Embedding(n_actions, embedding_dim)   # discrete: embedding, not Linear
        self.return_emb = nn.Linear(1, embedding_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(3 * seq_len, embedding_dim, num_heads, attention_dropout, residual_dropout)
            for _ in range(num_layers)])
        self.action_head = nn.Linear(embedding_dim, n_actions)     # discrete: logits, no tanh
        self.seq_len, self.embedding_dim = seq_len, embedding_dim
        self.state_dim, self.n_actions, self.episode_len = state_dim, n_actions, episode_len
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, states, actions, returns_to_go, time_steps, padding_mask=None):
        # states [B, L, state_dim], actions [B, L] (long), returns_to_go [B, L], time_steps [B, L]
        batch_size, seq_len = states.shape[0], states.shape[1]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        # (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (torch.stack([returns_emb, state_emb, act_emb], dim=1)
                    .permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim))
        if padding_mask is not None:
            padding_mask = (torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                            .permute(0, 2, 1).reshape(batch_size, 3 * seq_len))
        out = self.emb_drop(self.emb_norm(sequence))
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
        out = self.out_norm(out)
        # predict actions only from state embeddings -> [B, L, n_actions]
        return self.action_head(out[:, 1::3])


# ----------------------------------------------------------------------------- data
class SequenceSampler:
    """CORL's SequenceDataset: trajectories sampled proportional to length, a random start,
    a window of seq_len padded at the end (padding is masked in the loss)."""

    def __init__(self, data, seq_len, reward_scale, episode_len):
        obs = data["observations"].reshape(len(data["rewards"]), -1)
        self.trajs = []
        for t in data["trajectories"]:
            if not t["complete"]:
                continue
            r = data["rewards"][t["start"]:t["end"]].astype(np.float32)
            self.trajs.append(dict(
                obs=obs[t["start"]:t["end"]].astype(np.float32),
                actions=data["actions"][t["start"]:t["end"]],
                rtg=(np.cumsum(r[::-1])[::-1] * reward_scale).astype(np.float32)))
        lens = np.array([len(t["actions"]) for t in self.trajs])
        self.p = lens / lens.sum()
        self.seq_len, self.episode_len = seq_len, episode_len
        print(f"[dt] {len(self.trajs)} complete trajectories, {lens.sum()} transitions")

    def sample(self, batch_size, device):
        L = self.seq_len
        S = np.zeros((batch_size, L, self.trajs[0]["obs"].shape[1]), np.float32)
        A = np.zeros((batch_size, L), np.int64)
        R = np.zeros((batch_size, L), np.float32)
        T = np.zeros((batch_size, L), np.int64)
        M = np.zeros((batch_size, L), np.float32)
        for i, ti in enumerate(np.random.choice(len(self.trajs), size=batch_size, p=self.p)):
            tr = self.trajs[ti]
            s = np.random.randint(0, len(tr["actions"]))
            n = min(L, len(tr["actions"]) - s)
            S[i, :n], A[i, :n], R[i, :n] = tr["obs"][s:s + n], tr["actions"][s:s + n], tr["rtg"][s:s + n]
            T[i] = np.minimum(np.arange(s, s + L), self.episode_len + L - 1)
            M[i, :n] = 1
        return [torch.as_tensor(x, device=device) for x in (S, A, R, T, M)]


class DTPolicy:
    """Autoregressive rollout of the DT (CORL's eval_rollout) as an act_fn for common.evaluate."""

    def __init__(self, model, target_return, reward_scale, device):
        self.model, self.target, self.scale, self.device = model, target_return, reward_scale, device
        self.reset()

    def reset(self):
        self.states, self.actions, self.rtgs = [], [], [self.target * self.scale]

    @torch.no_grad()
    def __call__(self, obs):
        self.states.append(np.asarray(obs, np.float32).reshape(-1))
        L, t = self.model.seq_len, len(self.states)
        lo = max(0, t - L)
        S = torch.as_tensor(np.stack(self.states[lo:]), device=self.device).unsqueeze(0)
        A = torch.as_tensor(self.actions[lo:] + [0], device=self.device).long().unsqueeze(0)  # last is dummy
        R = torch.as_tensor(self.rtgs[lo:t], device=self.device).float().unsqueeze(0)
        T = torch.clamp(torch.arange(lo, t, device=self.device), max=self.model.episode_len - 1).unsqueeze(0)
        return int(self.model(S, A, R, T)[0, -1].argmax().item())

    def observe(self, action, reward, next_obs):
        self.actions.append(int(action))
        self.rtgs.append(self.rtgs[-1] - reward * self.scale)


def resolve_target(name, refs):
    key = f"ref/{name}"
    return float(refs[key]) if key in refs else float(name)


def main(args: Args):
    common.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    data = common.load_dataset(args.dataset)
    env_id = common.env_id_of(data, args.env_id)
    env = common.make_env(env_id)
    logger = common.Logger(args, "dt", args.dataset)
    refs = common.compute_refs(data, args, env_id, device)

    if args.episode_len <= 0:
        args.episode_len = max(t["end"] - t["start"] for t in data["trajectories"] if t["complete"])
    sampler = SequenceSampler(data, args.seq_len, args.reward_scale, args.episode_len)
    model = DecisionTransformer(int(np.prod(env.observation_space.shape)), env.action_space.n, args.seq_len,
                                args.episode_len, args.embedding_dim, args.num_layers, args.num_heads,
                                args.attention_dropout, args.residual_dropout, args.embedding_dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda steps: min((steps + 1) / args.warmup_steps, 1))
    targets = {n: resolve_target(n, refs) for n in args.target_returns}
    print(f"[dt] {sum(p.numel() for p in model.parameters())} parameters, target returns {targets}")

    for step in range(1, args.total_updates + 1):
        states, actions, returns, time_steps, mask = sampler.sample(args.batch_size, device)
        logits = model(states, actions, returns, time_steps, padding_mask=~mask.bool())
        loss = F.cross_entropy(logits.reshape(-1, env.action_space.n), actions.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()
        optim.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optim.step()
        scheduler.step()
        if step % 1000 == 0:
            logger.log({"losses/dt_loss": loss.item(), "charts/learning_rate": scheduler.get_last_lr()[0]}, step)
        if step % args.eval_freq == 0 or step == args.total_updates:
            model.eval()
            results = {}
            for n, target in targets.items():
                pol = DTPolicy(model, target, args.reward_scale, device)
                results[n] = common.evaluate(pol, env_id, args.eval_episodes, args.eval_seed,
                                             args.eval_max_steps, reset_fn=pol.reset)
            model.train()
            first = args.target_returns[0]
            extra = {f"eval/return_target_{n}": float(r.mean()) for n, r in results.items()}
            logger.log_eval(step, results[first], refs, extra)
    logger.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
