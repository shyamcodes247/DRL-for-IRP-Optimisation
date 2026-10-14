"""Tests for the upper dead zone in the inventory fraction mapping."""

import numpy as np
import pytest
import torch

from conftest import TEST_INSTANCE_PATHS
from environment.irp_env import IRPEnv
from training.mtppo import MTPPO
from training.rollout_buffer import RolloutBuffer

LOC_DIM, LOOKBACK = 2, 3


def _build(**kwargs) -> MTPPO:
    return MTPPO(
        node_feature_dims={"critic": LOC_DIM + 6, "inventory": LOC_DIM + 3,
                           "routing": 2 * LOC_DIM + 2},
        history_dim=1 + 2 * LOOKBACK, global_feature_dim=2,
        gin_dims=[32, 32], mlp_dims=[32], embed_dim=LOC_DIM + 2 * LOOKBACK,
        loc_dim=LOC_DIM, device="cpu", **kwargs,
    )


def _env() -> IRPEnv:
    return IRPEnv(data_file_path=TEST_INSTANCE_PATHS[0], loc_dim=LOC_DIM,
                  lookback_window=LOOKBACK, product_price=None, penalty_factor=None)


# The worked example: a g = 2 retailer, r = 10, U = 20, I = 0, so the minimum
# is 10 and the headroom is 20.
G2 = dict(inventory=np.array([0.0]), demand=np.array([10.0]), max_capacity=np.array([20.0]))


def test_off_reproduces_the_current_mapping_exactly():
    """
    The guarantee that existing results still hold. Swept across the whole
    action range rather than spot-checked, because the two expressions differ
    only in one divisor and a wrong `top` would show up as a small error at
    most fractions, not an obvious break.
    """
    for a in np.linspace(0.0, 1.0, 101):
        fraction = np.array([a])
        default = MTPPO._delivery_from_fraction(fraction, **G2)
        explicit = MTPPO._delivery_from_fraction(fraction, **G2, upper_dead_zone=None)
        # 0.1 is the dead zone; 1.0 is the implicit top of the original ramp.
        expected_disc = np.clip((a - 0.1) / 0.9, 0.0, 1.0)
        expected = 10.0 + expected_disc * 10.0
        assert default == pytest.approx(expected)
        assert explicit == pytest.approx(expected)


def test_off_reproduces_the_previous_update_parameters_exactly():
    """Beyond the mapping: a full PPO update must leave identical weights."""
    torch.manual_seed(0)
    reference = _build()
    torch.manual_seed(0)
    explicit = _build(upper_dead_zone=None)

    for agent in (reference, explicit):
        torch.manual_seed(1234)
        np.random.seed(1234)
        buffer = RolloutBuffer()
        agent.collect_episode(_env(), buffer)
        torch.manual_seed(99)
        np.random.seed(99)
        agent.update(buffer, ppo_epochs=4)

    for module in ("critic", "inv_actor", "routing_actor"):
        for (name, a), (_, b) in zip(getattr(reference, module).named_parameters(),
                                     getattr(explicit, module).named_parameters()):
            assert torch.equal(a, b), f"{module}.{name} diverged with the option off"


def test_at_or_above_the_threshold_delivers_the_full_headroom():
    """
    The whole point: every fraction from the threshold upward must deliver
    U - I *exactly*, not approximately. Exactness matters because a g = 2
    retailer is skipped next period only if it ends holding exactly one
    period of demand -- a residual of 1e-5 units still triggers a visit.
    """
    for a in (0.9, 0.95, 0.999999, 1.0):
        q = MTPPO._delivery_from_fraction(np.array([a]), **G2, upper_dead_zone=0.9)
        assert q == pytest.approx(20.0, abs=0.0), f"fraction {a} did not fill exactly"

    # Just below the threshold still ramps, so the option is a threshold and
    # not a blanket "always fill".
    below = MTPPO._delivery_from_fraction(np.array([0.89]), **G2, upper_dead_zone=0.9)
    assert below < 20.0
    # ...and the ramp is rescaled to span [0.1, 0.9] rather than [0.1, 1.0].
    assert below == pytest.approx(10.0 + ((0.89 - 0.1) / 0.8) * 10.0)


def test_a_filled_g2_retailer_can_be_skipped_the_next_period():
    """
    Ties the arithmetic to the behaviour it is supposed to buy. After a full
    fill the retailer ends the period holding exactly one period of demand, so
    next period's minimum is 0 and it needs no visit. Under the old mapping the
    same fraction left a residual that forced one.
    """
    inventory, demand, capacity = 0.0, 10.0, 20.0

    filled = MTPPO._delivery_from_fraction(
        np.array([0.9]), np.array([inventory]), np.array([demand]),
        np.array([capacity]), upper_dead_zone=0.9,
    )[0]
    stock_after = inventory + filled - demand
    assert max(0.0, demand - stock_after) == pytest.approx(0.0, abs=0.0)

    partial = MTPPO._delivery_from_fraction(
        np.array([0.9]), np.array([inventory]), np.array([demand]), np.array([capacity]),
    )[0]
    stock_after_partial = inventory + partial - demand
    next_minimum = max(0.0, demand - stock_after_partial)
    assert next_minimum > 1e-6, "old mapping should leave a visit-triggering residual"
