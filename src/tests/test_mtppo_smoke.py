from pathlib import Path

import pytest
import torch

from conftest import TEST_INSTANCE_PATHS
from environment.irp_env import IRPEnv
from training.mtppo import MTPPO

LOC_DIM = 2
LOOKBACK_WINDOW = 3
GIN_DIMS = [32, 32]
MLP_DIMS = [32]


def _build_mtppo() -> MTPPO:
    node_feature_dims = {
        "critic": LOC_DIM + 6,
        "inventory": LOC_DIM + 3,
        "routing": 2 * LOC_DIM + 2,
    }
    history_dim = 1 + 2 * LOOKBACK_WINDOW
    embed_dim = LOC_DIM + 2 * LOOKBACK_WINDOW
    return MTPPO(
        node_feature_dims=node_feature_dims,
        history_dim=history_dim,
        global_feature_dim=2,
        gin_dims=GIN_DIMS,
        mlp_dims=MLP_DIMS,
        embed_dim=embed_dim,
        loc_dim=LOC_DIM,
        device="cpu",
    )


def _assert_all_finite(module: torch.nn.Module) -> None:
    for name, param in module.named_parameters():
        assert torch.isfinite(param).all(), f"non-finite value in parameter {name!r}"


@pytest.mark.parametrize("instance_path", TEST_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_mtppo_trains_without_nan(instance_path):
    """
    Regression test for two numerical-stability bugs hit while wiring up
    training on this benchmark data: raw (non-[0,1]) instance coordinates
    blowing up through the GIN's neighbour-sum aggregation
    (`MTPPO._normalize_location`), and an unclamped log-std in
    `InventoryActor.forward` letting a single gradient step send sigma to
    an extreme value. Both caused NaNs within the first couple of PPO
    updates; this runs a few real epochs and checks every network
    parameter is still finite afterwards.
    """
    env = IRPEnv(
        instance_path,
        loc_dim=LOC_DIM,
        lookback_window=LOOKBACK_WINDOW,
        product_price=20.0,
        penalty_factor=0.3,
    )
    mtppo = _build_mtppo()

    mtppo.train(
        envs=[env],
        num_epochs=10,
        log_every=0,
    )

    _assert_all_finite(mtppo.critic)
    _assert_all_finite(mtppo.inv_actor)
    _assert_all_finite(mtppo.routing_actor)


@pytest.mark.parametrize("instance_path", TEST_INSTANCE_PATHS, ids=lambda p: Path(p).stem)
def test_skipping_diagnostics_does_not_change_the_solution(instance_path):
    """
    `collect_diagnostics=False` exists so inference can be timed without the
    2-opt reference tours dominating the measurement (see
    `scripts/benchmark_inference.py`). It is only a valid thing to time if it
    runs the *same* policy: every cost figure must come out bit-identical,
    and only the diagnostic fields may degrade to NaN.
    """
    torch.manual_seed(0)
    mtppo = _build_mtppo()
    env = IRPEnv(
        data_file_path=instance_path,
        loc_dim=LOC_DIM,
        lookback_window=LOOKBACK_WINDOW,
        product_price=None,
        penalty_factor=None,
    )

    full = mtppo.evaluate_episode(env, collect_diagnostics=True)
    fast = mtppo.evaluate_episode(env, collect_diagnostics=False)

    for key in ("inv_cost", "vrp_distance", "routing_cost", "total_cost",
                "fill_rate", "stockout_count", "fill_fraction"):
        assert fast[key] == full[key], f"{key} changed when diagnostics were skipped"

    # The diagnostics themselves are gone, not silently zero -- a zero would
    # read as "a perfect tour" in the logs.
    for key in ("visits_per_period", "vrp_distance_2opt", "vrp_excess_ratio",
                "nn_rank", "total_cost_best_route"):
        assert fast[key] != fast[key], f"{key} should be NaN without diagnostics"


def test_route_log_without_diagnostics_is_refused():
    """A route log that silently came back empty would look like a policy that
    never visited anyone, so the contradiction is rejected outright."""
    mtppo = _build_mtppo()
    env = IRPEnv(
        data_file_path=TEST_INSTANCE_PATHS[0],
        loc_dim=LOC_DIM,
        lookback_window=LOOKBACK_WINDOW,
        product_price=None,
        penalty_factor=None,
    )
    with pytest.raises(ValueError, match="collect_diagnostics"):
        mtppo.evaluate_episode(env, route_log=[], collect_diagnostics=False)
