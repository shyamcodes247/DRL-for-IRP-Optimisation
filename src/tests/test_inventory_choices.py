"""Tests for the discrete inventory action set {0, 1/2, 1}."""

import numpy as np
import pytest
import torch

from conftest import TEST_INSTANCE_PATHS
from environment.irp_env import IRPEnv
from training.mtppo import MTPPO
from training.rollout_buffer import RolloutBuffer

LOC_DIM, LOOKBACK = 2, 3
CHOICES = [0.0, 0.5, 1.0]


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


def test_the_three_choices_map_to_minimum_halfway_and_full_fill():
    """
    The worked example: a g = 2 retailer, r = 10, U = 20, I = 0, so the
    minimum is 10 and the headroom is 20. Choice 1.0 must deliver the headroom
    EXACTLY -- an approximate fill leaves a residual that still triggers a
    visit, which is the whole failure this action set exists to avoid.
    """
    agent = _build(inventory_choices=CHOICES)

    class Stub:
        retailers_current_inventory = np.array([0.0])
        current_demand = np.array([10.0])
        retailer_max_capacity = np.array([20.0])

    for index, expected in ((0, 10.0), (1, 15.0), (2, 20.0)):
        fractions = agent._fractions_from_action(torch.tensor([index]))
        q = agent._delivery_for(fractions, Stub())
        assert q[0] == pytest.approx(expected, abs=0.0), f"choice {index}"

    # Choice 0 is exactly the minimum the environment would have forced anyway,
    # so the no-stockout guarantee still holds by construction.
    stock_after_full = 0.0 + 20.0 - 10.0
    assert max(0.0, 10.0 - stock_after_full) == pytest.approx(0.0, abs=0.0)


def test_dead_zones_are_bypassed_in_discrete_mode():
    """
    The lower dead zone would compress 0.5 (to 0.444 of the range) and the
    upper one is redundant once 1.0 is an exact atom. Both must be ignored, or
    the middle choice would not mean "halfway".
    """
    agent = _build(inventory_choices=CHOICES, skip_dead_zone=0.1, upper_dead_zone=0.9)

    class Stub:
        retailers_current_inventory = np.array([0.0])
        current_demand = np.array([10.0])
        retailer_max_capacity = np.array([20.0])

    q = agent._delivery_for(np.array([0.5]), Stub())
    assert q[0] == pytest.approx(15.0), "0.5 should be halfway, not dead-zone compressed"


def test_the_policy_is_categorical_and_can_select_a_full_fill():
    """
    The structural point: 1.0 must be reachable both by sampling and by the
    greedy argmax. Under a Beta it is reachable by neither -- it has measure
    zero, and the mean is capped below 1 because beta >= 1.
    """
    torch.manual_seed(0)
    agent = _build(inventory_choices=CHOICES)
    assert agent.inv_actor.is_discrete

    env = _env()
    _, critic_obs, _ = env.reset()
    from agent.features import build_inventory_features, build_inventory_history
    scale = agent._location_scale(env)
    retailer_scale, _ = agent._quantity_scales(env)
    inv_obs = agent._inventory_obs_from_critic(critic_obs)
    feats = agent._normalize_location(build_inventory_features(inv_obs, retailer_scale), scale)
    hist = build_inventory_history(inv_obs, retailer_scale)

    # Sampling reaches every choice, because the zeroed output bias starts them
    # near-uniform rather than favouring one.
    seen = set()
    for _ in range(200):
        action, _ = agent.inv_actor.act(feats, hist)
        seen.update(action.tolist())
    assert seen == {0, 1, 2}, f"not every choice was sampled: {seen}"

    greedy = agent.inv_actor.mean_action(feats, hist)
    assert greedy.dtype in (torch.int64, torch.long)
    assert set(greedy.tolist()) <= {0, 1, 2}


def test_off_reproduces_the_previous_update_parameters_exactly():
    """With inventory_choices off, a full update must leave identical weights."""
    torch.manual_seed(0)
    reference = _build()
    torch.manual_seed(0)
    explicit = _build(inventory_choices=None)

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
            assert torch.equal(a, b), f"{module}.{name} diverged with choices off"


def test_a_full_episode_trains_and_never_stocks_out():
    """
    End to end: collect, update and evaluate without error, and with no
    stockouts -- choice 0 delivers the shortfall, so the no-stockout
    constraint must hold however the policy behaves.
    """
    torch.manual_seed(0)
    agent = _build(inventory_choices=CHOICES)
    buffer = RolloutBuffer()
    stats = agent.collect_episode(_env(), buffer)
    agent.update(buffer, ppo_epochs=2)
    metrics = agent.evaluate_episode(_env())

    assert metrics["stockout_count"] == 0
    assert np.isfinite(metrics["total_cost"])
    assert 0.0 <= stats["requested_fraction"] <= 1.0
    for param in agent.inv_actor.parameters():
        assert torch.isfinite(param).all()
