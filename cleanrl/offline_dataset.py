"""Save an online agent's replay buffer + model as an offline (D4RL-format) dataset.

Used for the fixed-data interventions (rebuttal): train BC / return-conditioned BC /
offline RL on exactly the experience D the online agent had at a checkpoint, and check
how much of the exploitation gap  V^{pi*_D}(s0) - V^{pi^theta}(s0)  can be recovered
without collecting new experience. See cleanrl/offline/README.md.

Dataset file (``<dir>/step_<global_step>.npz``) keys, D4RL style, in time order:
    observations, next_observations, actions (int64), rewards (extrinsic only),
    terminals, timeouts, online_returns (episodic returns of the last
    `return_buffer_size` online episodes, as used by BufferGapV2), and
    starts_mid_episode (True when the circular buffer had wrapped, so the first
    trajectory of each env is truncated at its start).
Atari (frame-stacked uint8 observations, memory-optimised SB3 buffer): only the newest frame of
each step is stored, as ``frames`` (N, H, W), with ``frame_stack`` = 4; no observations or
next_observations. cleanrl/offline/common.py rebuilds the stacks from the episode boundaries
(FrameStack repeats the first frame of an episode). Saves ~4x (x2 for next_observations) the space.
Model file: ``<dir>/step_<global_step>_model.pt`` with the state_dict and the args needed
to rebuild the network and the environment.
"""
import json
import os
from dataclasses import asdict, is_dataclass

import numpy as np
import torch


class TimeoutTracker:
    """SB3's ReplayBuffer (handle_timeout_termination=False) drops truncation flags,
    which are needed to split the buffer into trajectories. Call `record` right
    before `rb.add` with the same step's truncations."""

    def __init__(self, rb):
        self.rb = rb
        self.timeouts = np.zeros((rb.buffer_size, rb.n_envs), dtype=bool)

    def record(self, truncations):
        self.timeouts[self.rb.pos] = np.asarray(truncations, dtype=bool)


def _time_ordered(rb, arr):
    """Unroll the circular buffer into time order and stack envs one after another:
    (buffer_size, n_envs, ...) -> (n_envs * T, ...)."""
    if rb.full:
        idx = np.r_[rb.pos:rb.buffer_size, 0:rb.pos]
    else:
        idx = np.arange(rb.pos)
    a = arr[idx]
    a = np.swapaxes(a, 0, 1)
    return a.reshape((-1,) + a.shape[2:])


def save_replay_dataset(save_dir, global_step, rb, timeout_tracker, model, args, online_returns, extra=None):
    os.makedirs(save_dir, exist_ok=True)
    T = rb.buffer_size if rb.full else rb.pos
    terminals = _time_ordered(rb, rb.dones).astype(bool)
    timeouts = _time_ordered(rb, timeout_tracker.timeouts).copy()
    # The last transition of each env's chunk is cut off by the checkpoint, not by the
    # environment: mark it as a timeout so trajectories never run across env chunks.
    for e in range(rb.n_envs):
        end = (e + 1) * T - 1
        if not terminals[end]:
            timeouts[end] = True
    frame_stacked = rb.observations.ndim == 5  # (buffer, n_envs, stack, H, W)
    if frame_stacked:
        obs_data = dict(frames=_time_ordered(rb, rb.observations[:, :, -1]),
                        frame_stack=np.array(rb.observations.shape[2]))
    else:
        if getattr(rb, "optimize_memory_usage", False):
            next_obs = np.roll(rb.observations, -1, axis=0)  # SB3 stores next_obs in the following row
        else:
            next_obs = rb.next_observations
        obs_data = dict(observations=_time_ordered(rb, rb.observations),
                        next_observations=_time_ordered(rb, next_obs))
    data = dict(
        **obs_data,
        actions=_time_ordered(rb, rb.actions).reshape(-1).astype(np.int64),
        rewards=_time_ordered(rb, rb.rewards).astype(np.float32),
        terminals=terminals,
        timeouts=timeouts,
        online_returns=np.asarray(list(online_returns), dtype=np.float32),
        starts_mid_episode=np.array(rb.full),
        n_envs=np.array(rb.n_envs),
    )
    path = os.path.join(save_dir, f"step_{global_step}.npz")
    np.savez_compressed(path, **data)
    meta = {
        "global_step": int(global_step),
        "args": asdict(args) if is_dataclass(args) else dict(vars(args)),
        "num_transitions": int(len(data["rewards"])),
    }
    if extra:
        meta.update(extra)
    torch.save({"state_dict": model.state_dict(), "meta": meta},
               os.path.join(save_dir, f"step_{global_step}_model.pt"))
    with open(os.path.join(save_dir, f"step_{global_step}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[offline_dataset] saved {len(data['rewards'])} transitions to {path}")
    return path
