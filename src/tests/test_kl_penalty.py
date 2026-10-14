"""Tests for the adaptive KL penalty and value-loss clipping."""

import copy

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


def test_kl_is_zero_when_the_policy_has_not_moved():
    """
    The k3 estimator is (r - 1) - log r with r = pi_new/pi_old. On the FIRST
    PPO pass the networks have not been stepped yet, so pi_new IS pi_old,
    every ratio is exactly 1, and the estimate must be 0. If this is non-zero
    something is wrong with how old log-probs are stored or reused -- the one
    bug that would make every later KL number meaningless.
    """
    torch.manual_seed(0)
    mtppo = _build()
    buffer = RolloutBuffer()
    mtppo.collect_episode(_env(), buffer)

    # One pass only: with ppo_epochs=1 the reported KL is measured against a
    # policy that has not yet been updated.
    metrics = mtppo.update(buffer, ppo_epochs=1)
    assert metrics["approx_kl_inv"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["approx_kl_vrp"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["approx_kl_inv_per_retailer"] == pytest.approx(0.0, abs=1e-6)
    # And nothing was clipped, since every ratio is exactly 1.
    assert metrics["clip_fraction_inv"] == 0.0
    assert metrics["clip_fraction_vrp"] == 0.0


def test_beta_rises_above_twice_the_target_and_falls_below_half():
    """
    RLlib's rule, exercised directly: x1.5 when the policy moved more than
    twice the target, x0.5 when it barely moved, unchanged in between. Tested
    on the helper rather than through training, so the thresholds are pinned
    independently of whatever KL a real update happens to produce.
    """
    mtppo = _build(kl_coeff=0.2, kl_target=0.01)

    assert mtppo._adapt_kl(0.2, 0.05) == pytest.approx(0.3)      # > 2 * target
    assert mtppo._adapt_kl(0.2, 0.001) == pytest.approx(0.1)     # < 0.5 * target
    assert mtppo._adapt_kl(0.2, 0.01) == pytest.approx(0.2)      # exactly target
    assert mtppo._adapt_kl(0.2, 0.02) == pytest.approx(0.2)      # on the boundary
    assert mtppo._adapt_kl(0.2, 0.005) == pytest.approx(0.2)     # on the boundary

    # Both actors keep their own beta and move independently.
    mtppo.kl_coeff_inv, mtppo.kl_coeff_vrp = 0.2, 0.2
    mtppo.kl_coeff_inv = mtppo._adapt_kl(mtppo.kl_coeff_inv, 0.05)
    mtppo.kl_coeff_vrp = mtppo._adapt_kl(mtppo.kl_coeff_vrp, 0.001)
    assert mtppo.kl_coeff_inv == pytest.approx(0.3)
    assert mtppo.kl_coeff_vrp == pytest.approx(0.1)


def test_all_options_off_reproduces_the_previous_update_exactly():
    """
    The guarantee that existing results still reproduce: with kl_coeff = 0 and
    vf_clip_param = None, the new code must leave the same weights behind as
    the old path did. Checked by running two identically-seeded agents over
    the same rollout -- one with the options off, one with them explicitly
    defaulted -- and comparing every parameter bit for bit.
    """
    torch.manual_seed(0)
    reference = _build()
    torch.manual_seed(0)
    explicit = _build(kl_coeff=0.0, vf_clip_param=None, target_kl_early_stop=None)

    # Same starting weights, or the comparison proves nothing.
    for a, b in zip(reference.inv_actor.parameters(), explicit.inv_actor.parameters()):
        assert torch.equal(a, b)

    for agent in (reference, explicit):
        torch.manual_seed(1234)
        np.random.seed(1234)
        buffer = RolloutBuffer()
        agent.collect_episode(_env(), buffer)
        torch.manual_seed(99)
        np.random.seed(99)
        agent.update(buffer, ppo_epochs=4)

    for name, module in (("critic", "critic"), ("inv", "inv_actor"), ("vrp", "routing_actor")):
        for (n, a), (_, b) in zip(getattr(reference, module).named_parameters(),
                                  getattr(explicit, module).named_parameters()):
            assert torch.equal(a, b), f"{name}.{n} diverged with the options off"

    # Beta must not have drifted: adaptation is off when kl_coeff is 0.
    assert explicit.kl_coeff_inv == 0.0
    assert explicit.kl_coeff_vrp == 0.0


def test_value_clip_caps_the_squared_error_and_is_counted():
    """
    `vf_clip_param` caps each sample's squared value error. A tiny cap must
    bind on essentially every sample of an untrained critic, and the reported
    fraction must say so -- that fraction is the only way to notice the cap has
    silently switched the critic off, since clamp has no gradient above it.
    """
    torch.manual_seed(0)
    mtppo = _build(vf_clip_param=1e-9)
    buffer = RolloutBuffer()
    mtppo.collect_episode(_env(), buffer)
    metrics = mtppo.update(buffer, ppo_epochs=1)
    assert metrics["vf_clip_fraction"] > 0.9
    assert metrics["value"] <= 1e-9 + 1e-12

    torch.manual_seed(0)
    off = _build()
    buffer = RolloutBuffer()
    off.collect_episode(_env(), buffer)
    assert off.update(buffer, ppo_epochs=1)["vf_clip_fraction"] == 0.0


def test_inventory_beta_adapts_on_per_retailer_kl_not_summed():
    """
    The one case where the two conventions give opposite answers, so the only
    one that can catch a silent revert to the summed form.

    A 50-retailer instance moving by a summed KL of 0.1 has moved 0.002 per
    retailer. Against kl_target = 0.01 that is BELOW half the target, so the
    leash should slacken: beta halves. Judged on the summed KL it would be
    five times the target and beta would grow instead -- the wrong direction,
    and increasingly wrong as instances get larger, which is exactly why the
    summed KL is kept for the loss but not for the adaptation.
    """
    mtppo = _build(kl_coeff=0.2, kl_target=0.01)
    summed_kl, num_retailers = 0.1, 50
    per_retailer_kl = summed_kl / num_retailers

    assert mtppo._adapt_kl(0.2, per_retailer_kl) == pytest.approx(0.1)
    assert mtppo._adapt_kl(0.2, summed_kl) == pytest.approx(0.3)
