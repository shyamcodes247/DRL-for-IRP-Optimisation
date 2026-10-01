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
