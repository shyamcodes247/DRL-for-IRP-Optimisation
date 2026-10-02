import numpy as np
import pytest

from training.routing_diagnostics import (
    RouteRecorder,
    aggregate,
    distance_matrix,
    nearest_neighbour_tour,
    self_critical_advantage,
    tour_length,
    two_opt_tour,
)

# Depot at the origin with four retailers on a unit square, so the optimal
# tour is the perimeter (length 4) and any crossing order is strictly worse.
SQUARE = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 50.0]])


def test_distance_matrix_floors_like_archetti():
    """
    Archetti et al. (2007) define c_ij = floor(sqrt(dx^2 + dy^2)), and their
    published optima are computed against that. A 3-4-5 triangle cannot tell
    floor from round, so this pins a case where they differ.
    """
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 5.9]])
    dist = distance_matrix(coords)
    assert dist[0, 1] == 5.0                      # exact: floor == round
    assert dist[0, 2] == 5.0                      # 5.9 floors to 5, would round to 6
    assert dist[0, 0] == 0.0
    assert np.allclose(dist, dist.T)


def test_distance_matrix_agrees_with_the_environment():
    """
    The reference tours are only a fair yardstick if they measure distance the
    same way the environment charges it.
    """
    from environment.irp_env import IRPEnv

    from conftest import TEST_INSTANCE_PATHS

    env = IRPEnv(TEST_INSTANCE_PATHS[0], loc_dim=2, lookback_window=3,
                 product_price=None, penalty_factor=None)
    coords = np.vstack([env.depot_location, env.location])
    dist = distance_matrix(coords)
    for i in range(len(coords)):
        for j in range(len(coords)):
            assert dist[i, j] == env._get_distance(coords[i], coords[j])


def test_tour_length_is_closed():
    dist = distance_matrix(np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]]))
    # depot -> 1 -> 2 -> depot = 10 + 10 + floor(sqrt(200))
    assert tour_length(dist, [1, 2]) == 10 + 10 + np.floor(np.sqrt(200))
    assert tour_length(dist, []) == 0.0


def test_two_opt_is_never_worse_than_nearest_neighbour():
    dist = distance_matrix(SQUARE)
    nodes = [1, 2, 3, 4]
    nn = tour_length(dist, nearest_neighbour_tour(dist, nodes))
    opt = tour_length(dist, two_opt_tour(dist, nodes))
    assert opt <= nn


def test_nn_rank_is_zero_when_always_choosing_the_nearest():
    """A recorder fed the closest eligible node every hop should score 0."""
    dist = distance_matrix(SQUARE)
    rec = RouteRecorder(SQUARE, delivery_cost=1.0)
    position, eligible = 0, [1, 2, 3, 4]
    while eligible:
        nearest = min(eligible, key=lambda n: dist[position, n])
        rec.record_hop(position, np.array(eligible), nearest, -dist[position, nearest])
        eligible.remove(nearest)
        position = nearest
    rec.close_period()
    assert rec.periods[0]["nn_rank"] == 0.0
    assert rec.periods[0]["visits"] == 4


def test_nn_rank_is_one_when_always_choosing_the_farthest():
    dist = distance_matrix(SQUARE)
    rec = RouteRecorder(SQUARE, delivery_cost=1.0)
    position, eligible = 0, [1, 2, 3, 4]
    while eligible:
        farthest = max(eligible, key=lambda n: dist[position, n])
        rec.record_hop(position, np.array(eligible), farthest, -dist[position, farthest])
        eligible.remove(farthest)
        position = farthest
    rec.close_period()
    assert rec.periods[0]["nn_rank"] == 1.0


def test_excess_ratio_detects_a_bad_ordering():
    """A crossing order over the square must be longer than the 2-opt tour."""
    dist = distance_matrix(SQUARE)
    rec = RouteRecorder(SQUARE, delivery_cost=1.0)
    position = 0
    for node in [2, 4, 3, 1]:                      # deliberately crossing
        rec.record_hop(position, np.array([1, 2, 3, 4]), node, -dist[position, node])
        position = node
    rec.record_hop(position, np.array([]), 0, -dist[position, 0])
    rec.close_period(reference="two_opt")
    assert rec.periods[0]["ratio"] > 1.0


def test_delivery_cost_is_divided_out():
    """Travelled distance is reported raw, not scaled by delivery_cost."""
    rec = RouteRecorder(SQUARE, delivery_cost=5.0)
    rec.record_hop(0, np.array([1]), 1, -500.0)
    rec.close_period()
    assert rec.periods[0]["travelled"] == 100.0


def test_self_critical_advantage_signs():
    """Beating the reference scores positive, losing to it negative."""
    assert self_critical_advantage(travelled=80.0, reference=100.0, num_stops=5) == 0.2
    assert self_critical_advantage(travelled=120.0, reference=100.0, num_stops=5) == -0.2
    assert self_critical_advantage(travelled=100.0, reference=100.0, num_stops=5) == 0.0


def test_self_critical_advantage_is_zero_without_an_ordering_decision():
    """One stop has no ordering to get right, so it carries no signal."""
    assert self_critical_advantage(travelled=50.0, reference=100.0, num_stops=1) == 0.0
    assert self_critical_advantage(travelled=50.0, reference=0.0, num_stops=5) == 0.0


def test_recorder_exposes_the_advantage_for_the_period_just_closed():
    dist = distance_matrix(SQUARE)
    rec = RouteRecorder(SQUARE, delivery_cost=1.0)
    position = 0
    for node in [2, 4, 3, 1]:                      # deliberately crossing
        rec.record_hop(position, np.array([1, 2, 3, 4]), node, -dist[position, node])
        position = node
    rec.record_hop(position, np.array([]), 0, -dist[position, 0])
    rec.close_period()
    # A crossing order loses to nearest-neighbour, so the score is negative.
    assert rec.last_advantage() < 0.0
    assert rec.last_advantage() == rec.periods[-1]["advantage"]


def test_last_advantage_is_zero_before_any_period_closes():
    assert RouteRecorder(SQUARE, delivery_cost=1.0).last_advantage() == 0.0


def test_logit_spread_is_averaged_over_hops():
    rec = RouteRecorder(SQUARE, delivery_cost=1.0)
    rec.record_hop(0, np.array([1, 2]), 1, -100.0, logit_spread=0.2)
    rec.record_hop(1, np.array([2]), 2, -100.0, logit_spread=0.4)
    rec.close_period()
    assert rec.periods[0]["logit_spread"] == pytest.approx(0.3)


def test_summary_and_aggregate_skip_empty_periods():
    rec = RouteRecorder(SQUARE, delivery_cost=1.0)
    rec.close_period()                              # a period nobody needed serving
    rec.record_hop(0, np.array([1, 2]), 1, -100.0)
    rec.close_period()
    summary = rec.summary()
    assert summary["visits_per_period"] == 1.0      # the empty period is excluded
    assert summary["total_distance"] == 100.0

    merged = aggregate([summary, {}, rec.summary()])
    assert merged["mean_visits_per_period"] == 1.0
