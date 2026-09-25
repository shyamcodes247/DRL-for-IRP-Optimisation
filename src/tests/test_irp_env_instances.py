from pathlib import Path

import numpy as np
import pytest

from conftest import TEST_INSTANCE_PATHS
from environment.irp_env import IRPEnv

LOC_DIM = 2
LOOKBACK_WINDOW = 3


def _build_env(instance_path: str) -> IRPEnv:
    return IRPEnv(
        instance_path,
        loc_dim=LOC_DIM,
        lookback_window=LOOKBACK_WINDOW,
        product_price=20.0,
        penalty_factor=0.3,
    )


def _run_episode(env: IRPEnv, rng: np.random.Generator) -> int:
    """
    Drives one full episode with random (but always-eligible) actions,
    checking the invariants `IRPEnv` is meant to guarantee along the way.
    Returns the number of timesteps completed.
    """
    env.reset()
    terminated = False
    steps = 0

    while not terminated:
        action = env.inventory_action_space.sample()
        routing_obs, _, _ = env.inventory_action_step(action)

        assert (env.retailers_current_inventory >= 0).all()
        assert (env.retailers_current_inventory <= env.retailer_max_capacity).all()
        assert env.depot_inventory >= 0

        guard = 0
        while True:
            mask = routing_obs["visited_mask"]
            eligible = np.flatnonzero(mask == 0)
            assert len(eligible) > 0, "routing deadlock: no eligible node to visit"

            action = int(rng.choice(eligible))
            routing_obs, _, critic_obs, terminated, _, _ = env.routing_action_step(action)

            assert 0 <= env.current_load_capacity[0] <= env.vehicle_capacity

            if critic_obs is not None:
                break
            guard += 1
            assert guard < 1000, "routing loop did not terminate"

        steps += 1

    return steps


@pytest.mark.parametrize("instance_path", TEST_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_env_full_episode_invariants(instance_path):
    env = _build_env(instance_path)
    rng = np.random.default_rng(0)

    steps = _run_episode(env, rng)

    assert steps == env.episode_length


@pytest.mark.parametrize("instance_path", TEST_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_env_reset_is_reusable(instance_path):
    """Two consecutive episodes on the same env instance should both behave correctly."""
    env = _build_env(instance_path)
    rng = np.random.default_rng(1)

    first = _run_episode(env, rng)
    second = _run_episode(env, rng)

    assert first == second == env.episode_length


@pytest.mark.parametrize("instance_path", TEST_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_no_sales_loss_cost_enforces_hard_feasibility(instance_path):
    """
    Regression test: with product_price/penalty_factor both None (matching
    Archetti et al. (2007)'s objective, which has no sales-loss term),
    IRPEnv must enforce zero stockouts itself — since without an equivalent
    hard constraint, holding cost alone makes "replenish nothing, ever" the
    trivial reward-optimal policy (verified in this project to collapse
    training to zero routing activity). Requesting zero replenishment every
    period should still produce zero lost sales throughout the episode.
    """
    env = IRPEnv(
        instance_path,
        loc_dim=LOC_DIM,
        lookback_window=LOOKBACK_WINDOW,
        product_price=None,
        penalty_factor=None,
    )
    env.reset()
    zero_action = np.zeros(env.num_retailers, dtype=np.float32)
    terminated = False

    while not terminated:
        routing_obs, _, info = env.inventory_action_step(zero_action)
        assert info["lost_sales_units"] == 0.0

        guard = 0
        while True:
            eligible = np.flatnonzero(routing_obs["visited_mask"] == 0)
            assert len(eligible) > 0
            action = int(eligible[0])
            routing_obs, _, critic_obs, terminated, _, _ = env.routing_action_step(action)
            if critic_obs is not None:
                break
            guard += 1
            assert guard < 1000
