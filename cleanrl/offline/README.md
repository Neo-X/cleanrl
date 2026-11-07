# Fixed-data interventions (discrete actions)

Can the online agent's exploitation gap be recovered **without collecting new experience**?
We freeze the agent's replay buffer `D` at a checkpoint and train only a new learning
procedure on it, then compare against the online agent and the gap's reference
`V^{π̂*_D}(s0)` (the mean return of the top-k% complete episodes in `D`).

The algorithms are adapted to discrete actions from [CORL](https://github.com/corl-team/CORL)
(Apache-2.0), whose implementations are continuous-only. They run in this repo's
environment (gymnasium, torch 2), with no d4rl or mujoco-py dependency.

| Script | Intervention | CORL source | Discrete-action change |
|---|---|---|---|
| `bc_discrete.py --frac 1.0` | BC on all of `D` | `any_percent_bc.py` | cross-entropy on logits |
| `bc_discrete.py --frac 0.1` | BC on the top 10% trajectories | `any_percent_bc.py` | + rank complete episodes by undiscounted return |
| `dt_discrete.py` | Return-conditioned BC (Decision Transformer), conditioned on `topk_D` and `max_D` | `dt.py` | action `nn.Embedding`, logits head, clamped timesteps |
| `cql_discrete.py --cql_alpha 1` | Offline RL (CQL) from scratch | `cql.py` (not portable) | CQL(H), exact logsumexp over actions (Kumar et al. 2020), on the `dqn.py` update |
| `cql_discrete.py --cql_alpha 0` | Offline DQN from scratch | — | — |
| `cql_discrete.py --cql_alpha 0 --init_from_checkpoint` | Continue the online agent, standard replay, no new data | — | — |
| `... --top_frac 0.1 --top_prob 0.5` | Same, oversampling the top-10% trajectories | — | — |

Networks and DQN hyperparameters match `cleanrl/dqn.py`, so the online checkpoint can be loaded.

## 1. Collect datasets

`cleanrl/dqn.py --save_dataset_fractions 0.25 0.5 0.75 1.0` saves, at each fraction of
`total_timesteps`, to `<log_dir>datasets/<run_name>/`:

- `step_N.npz` holds the replay buffer in time order (D4RL keys: `observations`,
  `next_observations`, `actions`, `rewards` (extrinsic only), `terminals`, `timeouts`). It
  also holds the last `return_buffer_size` online episodic returns (`online_returns`, the
  window `BufferGapV2` logs).
- `step_N_model.pt` and `step_N.json` hold the Q-network and the run's args.

Only complete episodes are used for returns. A trajectory overwritten at the start of the
circular buffer, or cut off by the checkpoint, is still used for transitions (BC-100%, CQL),
but not for ranking or DT.

## 2. Train offline

```bash
python cleanrl/offline/cql_discrete.py --dataset <dir>/step_10000000.npz
python cleanrl/offline/bc_discrete.py  --dataset <dir>/step_10000000.npz --frac 0.1
python cleanrl/offline/dt_discrete.py  --dataset <dir>/step_10000000.npz
```

On a cluster (Mila, Narval or Rorqual) with [xm-slurm](../../../xm-slurm), from the repo root. The
image is built from the repo `Dockerfile` (the local working tree, uncommitted changes included):

```bash
../xm-slurm/.venv/bin/python launch_xm.py collect --cluster narval   # 5 DQN seeds, saves datasets
../xm-slurm/.venv/bin/python launch_xm.py offline --cluster narval   # all variants x 3 seeds on every dataset found
../xm-slurm/.venv/bin/python launch_xm.py offline --cluster narval --variants cql dt --steps 10000000 --dry_run
```

## Logged quantities (TensorBoard `runs/offline/`, wandb group `offline-<algo>`, `results.json`)

| Key | Meaning |
|---|---|
| `eval/return` | offline policy return (argmax; DT conditioned on the first `--target_returns`) |
| `ref/online_det` | online checkpoint, argmax Q, same evaluation protocol |
| `ref/topk_D` | `V^{π̂*_D}(s0)`: top-k% (default 5%) complete episodes in `D` |
| `ref/mean_D` | mean return of complete episodes in `D` (ε-greedy behaviour) |
| `ref/topk_online_window`, `ref/mean_online_window` | the online `BufferGapV2` window (last 1000 episodes) |
| `eval/gap_closed` | `(eval/return − ref/online_det) / (ref/topk_D − ref/online_det)` |

Note: `MinAtar/SpaceInvaders-v1` is fully deterministic (no RNG use in the game, sticky
actions off), so evaluation episodes are identical for a deterministic policy, and `topk_D`
cannot come from environment luck.
