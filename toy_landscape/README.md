# Toy landscape: studying RL local optima

A minimal, controllable environment for studying **when and why RL policies get
stuck in local optima**, and for visualising *returns over the state space* — the
`V(s)` sketch from the paper *"Is Exploration or Optimization the Problem for
Deep Reinforcement Learning?"* (see the repo `CLAUDE.md`).

The point of this toy is the **environment + visualization**, not new learners.
The RL algorithms being evaluated are the existing single-file implementations in
`cleanrl/` (PPO, DQN, SAC, PQN, ...). PPO is wired up; the others can be added
the same way.

## The environment

`Toy/Landscape-*-v0` — a 1D navigation task:

- **State**: continuous position `x ∈ [0, L]`, observed as `x / L ∈ [0, 1]` (a `Box`).
- **Action**: discrete `{0: left, 1: stay, 2: right}` (a `Discrete(3)`) — so it
  drops straight into `cleanrl/ppo.py`, which requires discrete actions.
- **Reward**: a *multimodal* `R(x)` = sum of Gaussian peaks of increasing height
  separated by low-reward valleys. The agent starts in the **lowest** peak's
  basin, so a greedy learner is tempted by the nearby local optimum while the
  global optimum sits across one or more valleys.
- **Episode**: fixed horizon (50 steps); the env emits standard Gymnasium
  `info["episode"]` stats so `buffer_gap`'s gap tracking works.

Reward presets (`env.py::REWARD_PRESETS`), selected via the env id:

| preset                          | shape                                              |
|---------------------------------|----------------------------------------------------|
| `Toy/Landscape-two_peaks-v0`    | low near peak + tall far peak (one valley)         |
| `Toy/Landscape-three_peaks-v0`  | three increasing peaks (matches the paper sketch)  |
| `Toy/Landscape-staircase-v0`    | four increasing peaks                              |
| `Toy/Landscape-deceptive-v0`    | wide easy local peak, narrow far global peak       |

`Toy/Landscape-v0` is an alias for `three_peaks`.

## Visualizing the gap (the sketch)

`visualize.py` plots `V(s)` over the state space with three markers:

- `π^θ`   — where the **learned** (greedy) policy ends up,
- `π̂*`   — the **experience-optimal** state: the best state the agent ever reached,
- `π*`    — the global optimum of the landscape.

A large `π̂* − π^θ` gap is an **exploitation/optimization** problem (the agent
generated good experience it can't reproduce); a large `π* − π̂*` gap is an
**exploration** problem (it never reached the good states).

## Usage

Quick, training-free look at a landscape (with reference `stay`/`right` policies):

```bash
.venv/bin/python toy_landscape/run.py --preset three_peaks
# -> toy_landscape/figures/three_peaks_reference.png
```

Train a real algorithm (PPO) on it — the hook in `ppo.py` writes the state-space
figure into the run's `log_dir` (and to TensorBoard/W&B) as training progresses:

```bash
.venv/bin/python cleanrl/ppo.py --env_id Toy/Landscape-three_peaks-v0 \
    --total_timesteps 500000 --num_envs 4 --num_steps 256 \
    --plot_freq 1 --log_dir runs/toy/
# -> runs/toy/landscape_three_peaks_<step>.png   (V(s) with π^θ, π̂*, π* markers)
# -> charts/* in TensorBoard: episodic_return, global/local optimality gap, etc.
```

## How it plugs into cleanrl

- `__init__.py` registers the `Toy/Landscape-*` envs with Gymnasium on import.
- `cleanrl/ppo.py` imports `toy_landscape` (registration) near the top and calls
  `toy_landscape.visualize.plot_ppo_landscape(...)` inside its periodic plotting
  block. Both are guarded/no-ops for non-toy envs, so normal runs are unaffected.
- The env emits a standard `info["episode"]` dict so `buffer_gap`'s
  best-experience (`π̂*`) and recent-best estimators are tracked automatically.

## Files

- `env.py`       — the Gymnasium env, reward presets, and registration.
- `visualize.py` — policy-agnostic state-space plotting + the `ppo.py` hook.
- `run.py`       — standalone landscape viewer (no training).
