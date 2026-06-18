# CLAUDE.md

This file orients Claude Code (and other contributors) to what this repository is for and how its pieces fit together.

## Purpose

This repo holds the research code for the paper **"Is Exploration or Optimization the Problem for Deep Reinforcement Learning?"** (Glen Berseth, Université de Montréal / Mila / CIFAR — NeurIPS 2026 submission). The LaTeX source lives at `../papers/RL-sub-optimality/main-neurips2026.tex`.

The paper's central question: when a deep RL agent fails to solve a hard task, is it because it **explores** badly (never generates good experience) or because it **exploits** badly (generates good experience but the deep network optimization can't learn from it)? The work argues the latter is the larger, under-appreciated culprit.

To diagnose this, the paper introduces a **practical sub-optimality** estimator (the `\methodName` macro in the paper). Instead of measuring how far the learned policy is from an unknown optimal policy `π*`, it measures the gap between the learned policy and an **experience optimal policy** `π̂*` — the best behavior the agent has *already* generated during training. A large gap means the agent is sitting on good experience it cannot reproduce → an exploitation/optimization problem, not an exploration one.

Key quantities (paper notation → meaning):
- **`policyAverage`** `V^{π̂^θ}(s₀)` — average return of the learned policy.
- **`topBestFive`** `V^{π̂*_{D∞}}(s₀)` — return of the top-k% (default **k = 5%**) of *all* trajectories ever generated ("best ever").
- **`topBestLocal`** `V^{π̂*_{D}}(s₀)` — return of the top-k% in the *current replay buffer* ("recent"), which the agent could in principle still train on.
- The **exploitation gap** = `topBest − policyAverage`. The headline finding: best generated experience is typically **2–3× better** than what the learned policy achieves.

## What's in here

This is a fork of [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL algorithm implementations — instrumented to track and plot the practical sub-optimality gap.

- **`cleanrl/`** — the single-file algorithm implementations. The ones used in the paper:
  - `ppo.py` (on-policy), `dqn.py` (value-based), `pqn.py` (parallelized DQN variant), `sac.py` (max-entropy, continuous control). These have been modified from upstream CleanRL to integrate gap tracking, RND intrinsic rewards (`--intrinsic_rewards RND`), and network scaling (`--num_layers`, ResNet vs. CNN, `--use_layer_norm`).
  - Other files (`c51*`, `td3*`, `ddpg*`, `rainbow_atari.py`, `*_jax.py`, etc.) are mostly upstream CleanRL and not central to the paper.
- **`cleanrl/buffer_gap.py`** — the core contribution code. Wrappers (`RecordEpisodeStatisticsV2`, `SyncVectorEnvV2`) and the `BufferGap` / `BufferGapV2` classes that track every reward/return/trajectory and compute the `topBestFive` (best-ever) and `topBestLocal` (recent) estimators during training. This is the "wrapper" the paper's Implementation Details section describes. The default top percentage is `0.05` (`top_return_buff_percentage` arg / `_gap_percentage`).
- **`plotting/`** — scripts that turn logged runs into the paper's figures: per-task gap plots (`make_plots_*`), exploration/RND comparisons (`make_plots_all_exploration.py`), scaling (`make_plots_all_scaling*.py`), per-algorithm `rliable` aggregate plots (`rliable_plot*.py`), and GRPO/RLVR results (`make_plots_GRPO.py`).
- **`run_jobs_list.sh`**, **`run_jobs.sh`** — SLURM (`sbatch`) launchers defining the experiment sweeps (environments × algorithms × seeds, plus RND and scaling variants). Experiments use **4–5 seeds**.
- **`toy_landscape/`** — a controllable 1D multimodal-reward toy env (`Toy/Landscape-*-v0`; discrete actions, continuous state) for studying when RL policies get stuck in local optima, plus state-space (`V(s)`) visualization of the exploitation gap (`π^θ` vs `π̂*` vs `π*`). Registers with Gymnasium and plugs into `cleanrl/ppo.py` via a guarded hook. See `toy_landscape/README.md`.
- **`data/`, `runs/`, `runs2/`** — logged experiment outputs.
- **`launch*.py`, `launch*.sh`, `cloud/`, `Dockerfile`** — cluster/Docker launch tooling.

## Environments studied

Difficult exploration/exploitation tasks where the gap shows up: MinAtar (SpaceInvaders, Breakout, Asterix), Atari / Atari-5 (Montezuma's Revenge, BattleZone, NameThisGame), Craftax, and Sokoban via RLVR (GRPO fine-tuning of Qwen2.5-1.5B-Instruct with LoRA). Easier continuous-control tasks used as low-gap controls: HalfCheetah, Walker2d, Humanoid.

## Experiment tracking

Runs log to Weights & Biases — project `sub-optimality`, entity `real-lab` (see `ppo.py` args). Pass `--track` to enable. TensorBoard summaries are also written.

## Key results the code supports

1. **Per-task** (`seq:results-per-task`): hard tasks show a large, persistent exploitation gap; easy/solved tasks (HalfCheetah, 1-layer) show none. The gap does not shrink with more training.
2. **Adding exploration** (`seq:results-exploration`): RND raises returns *and* widens the gap — better exploration makes exploitation harder.
3. **Scaling networks** (`seq:results-scaling`): bigger nets (ResNet-18 vs 3-layer CNN; 4→256 layers on HalfCheetah) widen the gap → scaling pain is an optimization/exploitation issue.
4. **Across algorithms** (`seq:results-per-alg`): aggregated with a modified `rliable` optimality-gap metric (using `topBest` as the upper bound instead of a heuristic human-level `r_max/(1−γ)`); PPO/DQN reach only ~30% of their best generated experience.

## Conventions / gotchas

- Returns are **undiscounted** sums of rewards, to match standard Atari/MuJoCo evaluation.
- The strict deterministic `π̂*` (replaying the single best action sequence, Eq. pi-max) is high-variance and only valid in deterministic envs; the **softer top-k% estimators** (`topBestFive`, `topBestLocal`) are what's used in practice.
- Deterministic-vs-stochastic policy evaluation matters: for PPO continuous use the policy mean; for discrete use `argmax_a Q(s,a)` (see paper's deterministic-envs appendix).
- This is a research fork; prefer the paper's `cleanrl/{ppo,dqn,pqn,sac}.py` + `buffer_gap.py` over upstream files when reasoning about the method.
