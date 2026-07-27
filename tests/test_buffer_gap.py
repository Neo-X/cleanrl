import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cleanrl"))
import buffer_gap  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests: k selection math (no env / training required)
# ---------------------------------------------------------------------------

def test_topk_mean_and_variance_matches_manual_computation():
    returns = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    v_k, var_k = buffer_gap.topk_mean_and_variance(returns, k=2)
    assert v_k == pytest.approx((10.0 + 4.0) / 2)
    assert var_k >= 0.0


def test_topk_mean_and_variance_k_equals_n_is_sample_mean():
    returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v_k, _ = buffer_gap.topk_mean_and_variance(returns, k=len(returns))
    assert v_k == pytest.approx(returns.mean())


def test_bootstrap_topk_variance_is_nonnegative_and_reproducible():
    rng = np.random.default_rng(0)
    returns = rng.normal(5.0, 2.0, size=500)
    v_a, var_a = buffer_gap.bootstrap_topk_variance(returns, k=25, n_bootstrap=50, rng=np.random.default_rng(1))
    v_b, var_b = buffer_gap.bootstrap_topk_variance(returns, k=25, n_bootstrap=50, rng=np.random.default_rng(1))
    assert var_a >= 0.0
    assert v_a == pytest.approx(v_b)
    assert var_a == pytest.approx(var_b)


@pytest.mark.parametrize("method", ["closed_form", "bootstrap"])
def test_compute_optimal_k_selects_smallest_k_meeting_precision(method):
    rng = np.random.default_rng(0)
    returns = rng.normal(5.0, 2.0, size=2000)
    result = buffer_gap.compute_optimal_k(returns, epsilon=0.10, method=method, n_bootstrap=50,
                                           rng=np.random.default_rng(0))
    assert result["found"]
    assert 1 <= result["k_star"] <= len(returns)
    assert result["alpha_star"] == pytest.approx(result["k_star"] / len(returns))
    # k just below k_star (if present in the grid) must fail the precision target,
    # otherwise k_star would not be the smallest satisfying k.
    ks = sorted(result["table"])
    idx = ks.index(result["k_star"])
    if idx > 0:
        assert result["table"][ks[idx - 1]]["rel_se"] > 0.10


def test_compute_optimal_k_falls_back_to_largest_k_when_unmet():
    # tiny, high-variance sample: an unreachably tight epsilon must fall back to the
    # largest k in the grid. Uses the bootstrap method: the closed-form estimator is
    # degenerate at k=1 (V_k == q there, so its variance term is trivially zero).
    returns = np.array([1.0, 100.0])
    result = buffer_gap.compute_optimal_k(returns, k_grid=[1, 2], epsilon=1e-6, method="bootstrap",
                                           n_bootstrap=50, rng=np.random.default_rng(0))
    assert not result["found"]
    assert result["k_star"] == 2


# ---------------------------------------------------------------------------
# Unit tests: BufferGapV2 bookkeeping
# ---------------------------------------------------------------------------

def make_episode(return_, rewards, actions=None, key="return", length_key="length"):
    info = {"rewards": np.asarray(rewards), "actions": np.asarray(actions if actions is not None else rewards)}
    info[key] = return_
    if length_key:
        info[length_key] = len(rewards)
    return info


def test_buffer_gap_v2_accepts_return_key():
    gs = buffer_gap.BufferGapV2(buffer_size=100, top_buffer_percet=0.5)
    gs.add(make_episode(3.0, [0, 0, 3], key="return", length_key="length"))
    assert list(gs._returns) == [3.0]


def test_buffer_gap_v2_accepts_r_key():
    """dqn_atari.py / ppo_continuous_action.py / sac_continuous_action.py chain the base
    gym.wrappers.RecordEpisodeStatistics before RecordEpisodeStatisticsV2, so their episode
    dict uses "r"/"l" instead of "return"/"length"."""
    gs = buffer_gap.BufferGapV2(buffer_size=100, top_buffer_percet=0.5)
    gs.add(make_episode(3.0, [0, 0, 3], key="r", length_key="l"))
    assert list(gs._returns) == [3.0]


def test_buffer_gap_v2_tracks_best_trajectory():
    gs = buffer_gap.BufferGapV2(buffer_size=100, top_buffer_percet=0.5)
    gs.add(make_episode(1.0, [1]))
    gs.add(make_episode(5.0, [5]))
    gs.add(make_episode(2.0, [2]))
    assert gs._max_return == 5.0
    assert list(gs._best_traj_r) == [5]


def test_buffer_gap_v2_compute_optimal_k_uses_stored_returns():
    gs = buffer_gap.BufferGapV2(buffer_size=1000, top_buffer_percet=0.5)
    rng = np.random.default_rng(0)
    for r in rng.normal(5.0, 2.0, size=500):
        gs.add(make_episode(float(r), [float(r)]))
    result = gs.compute_optimal_k(epsilon=0.10, method="closed_form")
    assert 1 <= result["k_star"] <= 500


# ---------------------------------------------------------------------------
# Integration smoke tests: each training script must run end-to-end without
# crashing, and must actually populate the optimality-gap diagnostics
# (not silently no-op due to a broken episode-info wrapper chain).
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _run(cmd):
    subprocess.run(cmd, shell=True, check=True, cwd=REPO_ROOT)


def _newest_scalar_tags(search_root):
    """Find the most recently modified tensorboard event file under search_root
    and return the set of scalar tags it contains."""
    import glob as _glob

    from tensorboard.backend.event_processing import event_accumulator

    event_files = _glob.glob(os.path.join(search_root, "**", "events.out.tfevents.*"), recursive=True)
    assert event_files, f"no tensorboard event file found under {search_root!r}"
    newest = max(event_files, key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(os.path.dirname(newest))
    ea.Reload()
    return ea.Tags()["scalars"]


def _assert_k_star_logged(search_root):
    tags = _newest_scalar_tags(search_root)
    assert "charts/k_star" in tags, "k_star was not logged: gap_stats._returns may never have been populated"
    assert "charts/alpha_star" in tags


def test_ppo_smoke(tmp_path):
    # num_envs * num_steps must exceed 500: ppo.py's "reward top 95%" logging does a
    # fixed torch.topk(..., 500) over the flattened rollout, independent of buffer_gap.
    log_dir = str(tmp_path) + os.sep
    _run(
        "python cleanrl/ppo.py --env-id MinAtar/Breakout-v0 --num-envs 4 --num-steps 128 "
        f"--total-timesteps 1024 --plot-freq 1 --return-buffer-size 50 --log-dir {log_dir}"
    )
    _assert_k_star_logged(log_dir)


def test_dqn_smoke(tmp_path):
    log_dir = str(tmp_path) + os.sep
    _run(
        "python cleanrl/dqn.py --env-id MinAtar/Breakout-v1 --total-timesteps 2000 "
        "--learning-starts 100 --buffer-size 1000 --return-buffer-size 50 --plot-freq 10 "
        f"--intrinsic-rewards \"\" --log-dir {log_dir}"
    )
    _assert_k_star_logged(log_dir)


def test_ppo_continuous_action_smoke(tmp_path):
    log_dir = str(tmp_path) + os.sep
    _run(
        "python cleanrl/ppo_continuous_action.py --env-id Hopper-v4 --num-envs 2 --num-steps 64 "
        f"--total-timesteps 1024 --plot-freq 1 --return-buffer-size 50 --log-dir {log_dir}"
    )
    _assert_k_star_logged(log_dir)


def test_sac_continuous_action_smoke():
    # sac_continuous_action.py hardcodes SummaryWriter(f"runs/{run_name}"), ignoring
    # --log-dir, so its output always lands under <repo>/cleanrl/runs/ regardless of cwd.
    _run(
        "python cleanrl/sac_continuous_action.py --env-id Hopper-v4 --total-timesteps 2000 "
        "--learning-starts 100 --buffer-size 1000 --return-buffer-size 50 --plot-freq 50"
    )
    _assert_k_star_logged(os.path.join(REPO_ROOT, "runs"))


@pytest.mark.skipif(
    subprocess.run("python -c \"import ale_py\"", shell=True, cwd=REPO_ROOT).returncode != 0,
    reason="ale-py / Atari ROMs not available",
)
def test_dqn_atari_smoke(tmp_path):
    log_dir = str(tmp_path) + os.sep
    _run(
        "python cleanrl/dqn_atari.py --env-id ALE/Pong-v5 --total-timesteps 2000 "
        "--learning-starts 100 --buffer-size 1000 --return-buffer-size 50 --plot-freq 10 "
        f"--intrinsic-rewards \"\" --log-dir {log_dir}"
    )
    _assert_k_star_logged(log_dir)
