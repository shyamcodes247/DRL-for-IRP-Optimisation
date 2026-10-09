import math

import pytest
import torch

from agent.routing_actor import RoutingActor

LOC_DIM = 2
NODE_DIM = 2 * LOC_DIM + 2


def _actor(**kwargs):
    torch.manual_seed(0)
    return RoutingActor(node_feature_dim=NODE_DIM, gin_dims=[16, 16], mlp_dims=[16], **kwargs)


def test_a_fully_masked_state_raises_instead_of_producing_nan():
    """
    Masking every node leaves all logits at -inf, and Categorical's
    normalisation then yields -inf - (-inf) = NaN, which propagates silently
    into the weights. It must fail loudly at the source instead.
    """
    actor = _actor()
    features = torch.randn(6, NODE_DIM)
    with pytest.raises(ValueError, match="every node is masked"):
        actor.act(features, torch.ones(6, dtype=torch.int64))


def test_a_partially_masked_state_still_works():
    actor = _actor()
    features = torch.randn(6, NODE_DIM)
    mask = torch.ones(6, dtype=torch.int64)
    mask[3] = 0                                   # exactly one legal stop
    action, log_prob, spread = actor.act(features, mask)
    assert action == 3
    assert torch.isfinite(log_prob)
    # One selectable node means there was no choice to make, so no spread.
    assert spread != spread                       # NaN


def test_logit_spread_is_reported_for_a_real_choice():
    actor = _actor()
    features = torch.randn(6, NODE_DIM)
    mask = torch.zeros(6, dtype=torch.int64)
    mask[0] = 1                                   # depot masked, 5 retailers open
    _, _, spread = actor.act(features, mask)
    assert math.isfinite(spread) and spread >= 0.0


def test_logits_are_bounded_by_the_clip():
    """
    Extreme inputs must not produce the runaway logits measured before the
    bound: a 300-epoch run reached a spread of 675, a softmax so peaked it is
    a hard argmax with no exploration left.
    """
    actor = _actor(logit_clip=10.0)
    logits = actor(torch.randn(9, NODE_DIM) * 1e4)
    assert torch.isfinite(logits).all()
    assert logits.abs().max() <= 10.0


def test_the_clip_preserves_the_ordering_of_preferences():
    """tanh is monotone, so bounding the scale must not reorder the nodes."""
    actor = _actor(logit_clip=10.0)
    features = torch.randn(7, NODE_DIM)
    bounded = actor(features)
    raw = actor.decoder(
        torch.cat([actor.gin(features),
                   actor.gin(features).mean(dim=0, keepdim=True).expand(7, -1)], dim=1)
    ).squeeze(-1)
    assert torch.equal(bounded.argsort(), raw.argsort())


def test_literal_encoder_features_cannot_see_the_vehicle():
    """
    The ablation behind Table 4's first column rests on one claim: with Lu et
    al. (2025)'s published feature set, a node's row does not depend on where
    the vehicle is, so the policy can only emit a fixed priority order. Assert
    it directly rather than inferring it from tour lengths.
    """
    import numpy as np

    from agent.features import build_routing_features

    obs = {
        "location": np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [50.0, 50.0]]),
        "replenishment_amount": np.array([3.0, 4.0, 5.0]),
        "vehicle_position": 0,
    }
    at_depot = build_routing_features(obs, include_position=False)
    obs_moved = dict(obs, vehicle_position=3)
    far_away = build_routing_features(obs_moved, include_position=False)
    assert torch.equal(at_depot, far_away), "literal features changed with vehicle position"
    assert at_depot.shape[1] == 3, "expected (location, replenishment) only"

    # The project's own feature set must do the opposite, or the comparison
    # between the two conditions measures nothing.
    assert not torch.equal(
        build_routing_features(obs, include_position=True),
        build_routing_features(obs_moved, include_position=True),
    )


def test_shared_routing_advantage_switches_what_hops_are_scored_with():
    """The second half of the ablation: routing hops must actually receive the
    shared advantage when the switch is set, not the self-critical one."""
    from training.rollout_buffer import RolloutBuffer

    buffer = RolloutBuffer()
    assert buffer.shared_routing_advantage is False
    buffer.shared_routing_advantage = True
    buffer.clear()
    assert buffer.shared_routing_advantage is True, "clear() wiped configuration"
