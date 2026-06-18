"""A 1D multimodal-reward toy environment for studying RL local optima.

The agent lives on a line ``x in [0, L]`` and at every step moves left, stays,
or moves right (a **discrete** 3-action space).  Its position is reported as a
**continuous** observation ``x / L in [0, 1]``.  The reward is a *multimodal*
function of position ``R(x)`` made of several Gaussian peaks of increasing
height separated by low-reward valleys (see ``REWARD_PRESETS``).  The agent
starts in the basin of the *lowest* peak, so a greedy learner is tempted to
climb the nearby local optimum while the global optimum sits across one or more
valleys.

This is the minimal setting that reproduces the sketch in the paper: plotting
``R(x)`` over the state space gives the multimodal ``V(s)`` curve, and different
algorithms converge to different peaks (``pi_theta`` local, ``pi*`` global).

The env implements the Gymnasium API (``Box`` observation, ``Discrete`` action)
so it drops straight into ``cleanrl/ppo.py`` via ``gym.make``.  It is also light
enough (numpy only in the core) to use directly in the standalone study scripts.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Each preset is a list of (center, width, height) Gaussian bumps.  Heights
# increase left-to-right so the global optimum is the right-most peak, and the
# agent starts at the left-most (lowest) peak.
REWARD_PRESETS: dict[str, list[tuple[float, float, float]]] = {
    # Classic local optimum: a low near peak and a tall far peak, one valley
    # between them that the agent must cross.
    "two_peaks": [(2.0, 0.6, 1.0), (8.0, 0.6, 2.2)],
    # Three increasing peaks -> matches the hand drawn figure (pi_theta, pi_hat*, pi*).
    "three_peaks": [(2.0, 0.6, 1.0), (5.0, 0.6, 1.6), (8.0, 0.6, 2.4)],
    # A staircase of four peaks: each step out of a basin is a small commitment.
    "staircase": [(1.5, 0.5, 0.8), (4.0, 0.5, 1.3), (6.5, 0.5, 1.8), (9.0, 0.5, 2.4)],
    # Deceptive: the local peak nearest the start is *wide* and easy to reach,
    # the global peak is narrow and far -> exploration is strongly punished.
    "deceptive": [(2.0, 1.2, 1.4), (8.5, 0.35, 2.6)],
}


def make_landscape(bumps: list[tuple[float, float, float]]) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorised reward function ``R(x)`` from a list of Gaussian bumps."""

    centers = np.array([b[0] for b in bumps])
    widths = np.array([b[1] for b in bumps])
    heights = np.array([b[2] for b in bumps])

    def reward(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)[..., None]
        bump = heights * np.exp(-((x - centers) ** 2) / (2.0 * widths**2))
        return bump.sum(axis=-1)

    return reward


class LandscapeEnv(gym.Env):
    """1D discrete-action navigation over a multimodal, continuous-state landscape.

    Discrete actions ``{0: left, 1: stay, 2: right}``; the observation is the
    scalar position normalised to ``[0, 1]``.  Extra keyword arguments (e.g. the
    MinAtar-specific ``sticky_action_prob`` / ``difficulty_ramping`` that
    ``cleanrl/ppo.py`` passes to ``gym.make``) are accepted and ignored.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        preset: str = "three_peaks",
        length: float = 10.0,
        horizon: int = 50,
        step_size: float = 0.3,
        start: float = 2.0,
        start_noise: float = 0.1,
        move_noise: float = 0.05,
        render_mode: str | None = None,
        **kwargs,  # absorb env-id-specific kwargs from gym.make (e.g. MinAtar's)
    ) -> None:
        super().__init__()
        if preset not in REWARD_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {list(REWARD_PRESETS)}")
        self.preset = preset
        self.length = float(length)
        self.horizon = int(horizon)
        self.step_size = float(step_size)
        self.start = float(start)
        self.start_noise = float(start_noise)
        self.move_noise = float(move_noise)
        self.render_mode = render_mode

        self.reward_fn = make_landscape(REWARD_PRESETS[preset])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # convenience attributes used by the standalone study scripts
        self.n_actions = 3
        self.obs_dim = 1
        self.x = self.start
        self.t = 0

    # -- Gymnasium API ----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.x = float(np.clip(self.start + self.np_random.normal(0, self.start_noise), 0, self.length))
        self.t = 0
        self._ep_return = 0.0
        self._ep_start = time.perf_counter()
        return self._obs(), {"x": self.x}

    def step(self, action: int):
        direction = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]
        self.x = float(
            np.clip(
                self.x + direction * self.step_size + self.np_random.normal(0, self.move_noise),
                0,
                self.length,
            )
        )
        self.t += 1
        reward = float(self.reward_fn(self.x))
        self._ep_return += reward
        terminated = False  # never terminates early; bounded by the horizon
        truncated = self.t >= self.horizon
        info = {"x": self.x}
        if terminated or truncated:
            # Emit the standard Gymnasium episode-statistics dict.  buffer_gap's
            # RecordEpisodeStatisticsV2 only *augments* this (adds actions/rewards),
            # it does not create it, so the env supplies it directly.
            info["episode"] = {
                "r": np.float32(self._ep_return),
                "l": np.int32(self.t),
                "t": np.float32(time.perf_counter() - self._ep_start),
            }
        return self._obs(), reward, terminated, truncated, info

    # -- helpers ----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.array([self.x / self.length], dtype=np.float32)

    def obs_to_x(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs)[..., 0] * self.length

    # -- analysis helpers (used only for plotting / oracles) --------------
    def landscape(self, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(0, self.length, n)
        return xs, self.reward_fn(xs)

    @property
    def global_optimum_x(self) -> float:
        xs, rs = self.landscape(2000)
        return float(xs[int(np.argmax(rs))])


def register_envs() -> None:
    """Register one Gymnasium id per reward preset, e.g. ``Toy/Landscape-three_peaks-v0``.

    Also registers ``Toy/Landscape-v0`` as an alias for the default preset so the
    env can be selected with ``--env_id Toy/Landscape-v0`` in ``cleanrl/ppo.py``.
    """
    horizon = 50
    for preset in REWARD_PRESETS:
        env_id = f"Toy/Landscape-{preset}-v0"
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point=LandscapeEnv,  # pass the class directly: import-path agnostic
            max_episode_steps=horizon,
            kwargs={"preset": preset, "horizon": horizon},
        )
    if "Toy/Landscape-v0" not in gym.registry:
        gym.register(
            id="Toy/Landscape-v0",
            entry_point=LandscapeEnv,
            max_episode_steps=horizon,
            kwargs={"preset": "three_peaks", "horizon": horizon},
        )
