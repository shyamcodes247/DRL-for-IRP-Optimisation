import numpy as np
import pytest
import torch

from agent.inventory_actor import InventoryActor
from training.mtppo import MTPPO

LOC_DIM, LOOKBACK = 2, 3


DEAD_ZONE = 0.1


def _delivery(fraction, inventory, demand, capacity, dead_zone=DEAD_ZONE):
    return MTPPO._delivery_from_fraction(
        np.asarray(fraction, dtype=float),
        np.asarray(inventory, dtype=float),
        np.asarray(demand, dtype=float),
        np.asarray(capacity, dtype=float),
        dead_zone,
    )


def test_fraction_zero_delivers_exactly_the_shortfall():
    """Fraction 0 must reproduce the env's own no-stockout floor."""
    # stock 30, demand 50 -> short by 20; capacity 100 -> headroom 70
    assert _delivery([0.0], [30.0], [50.0], [100.0]) == pytest.approx([20.0])


def test_fraction_one_tops_up_to_max_capacity():
    delivered = _delivery([1.0], [30.0], [50.0], [100.0])
    assert delivered == pytest.approx([70.0])          # 30 + 70 = U_i
    assert (30.0 + delivered[0]) == pytest.approx(100.0)


def test_fraction_interpolates_between_the_two():
    # 0.5 sits (0.5 - 0.1) / 0.9 = 4/9 of the way up the discretionary range
    assert _delivery([0.5], [30.0], [50.0], [100.0]) == pytest.approx([20.0 + 50.0 * 4 / 9])


def test_no_shortfall_means_the_whole_headroom_is_discretionary():
    """Stock already covers demand, so a low fraction delivers nothing at all."""
    assert _delivery([0.0], [80.0], [50.0], [100.0]) == pytest.approx([0.0])
    assert _delivery([1.0], [80.0], [50.0], [100.0]) == pytest.approx([20.0])


def test_dead_zone_makes_skipping_reachable():
    """A retailer needing nothing this period gets exactly zero, so it is not visited."""
    # stock 80 covers demand 50 -> no shortfall
    for fraction in (0.0, 0.05, 0.1):
        assert _delivery([fraction], [80.0], [50.0], [100.0]) == pytest.approx([0.0])
    # just above the dead zone it ramps from zero rather than jumping
    assert _delivery([0.1001], [80.0], [50.0], [100.0])[0] == pytest.approx(0.0, abs=1e-2)
    assert _delivery([0.55], [80.0], [50.0], [100.0])[0] > 0.0


def test_dead_zone_never_skips_a_retailer_that_would_stock_out():
    """Below the dead zone the shortfall is still delivered in full."""
    for fraction in (0.0, 0.05, 0.1):
        # stock 30 against demand 50 -> must still receive 20
        assert _delivery([fraction], [30.0], [50.0], [100.0]) == pytest.approx([20.0])


def test_delivery_is_continuous_across_the_dead_zone_edge():
    below = _delivery([0.0999], [80.0], [50.0], [100.0])[0]
    above = _delivery([0.1001], [80.0], [50.0], [100.0])[0]
    assert abs(above - below) < 1e-2


def test_no_stockout_holds_for_any_fraction():
    rng = np.random.default_rng(1)
    for _ in range(300):
        capacity = rng.uniform(10, 200, size=6)
        inventory = rng.uniform(0, 1, size=6) * capacity
        demand = rng.uniform(0, 1.5, size=6) * capacity
        delivered = _delivery(rng.uniform(0, 1, size=6), inventory, demand, capacity)
        # Either demand is met, or it was physically impossible to meet it
        # (the retailer cannot hold that much), never a policy choice.
        met = inventory + delivered + 1e-9 >= np.minimum(demand, capacity)
        assert met.all()


def test_delivery_never_exceeds_headroom_or_goes_negative():
    rng = np.random.default_rng(0)
    for _ in range(200):
        capacity = rng.uniform(10, 200, size=5)
        inventory = rng.uniform(0, 1, size=5) * capacity
        demand = rng.uniform(0, 1.5, size=5) * capacity
        fraction = rng.uniform(0, 1, size=5)
        delivered = _delivery(fraction, inventory, demand, capacity)
        assert (delivered >= -1e-9).all()
        assert (inventory + delivered <= capacity + 1e-9).all()


def test_full_inventory_leaves_no_room():
    """A retailer already at capacity can receive nothing, whatever is asked."""
    assert _delivery([1.0], [100.0], [50.0], [100.0]) == pytest.approx([0.0])


def test_inventory_actor_emits_fractions_in_the_unit_interval():
    torch.manual_seed(0)
    actor = InventoryActor(
        node_feature_dim=LOC_DIM + 3, history_dim=1 + 2 * LOOKBACK,
        gin_dims=[16, 16], mlp_dims=[16], embed_dim=8,
    )
    nodes = torch.randn(7, LOC_DIM + 3)
    history = torch.randn(7, 1 + 2 * LOOKBACK)

    action, log_prob = actor.act(nodes, history)
    assert action.shape == (7,)
    assert ((action > 0) & (action < 1)).all()
    assert torch.isfinite(log_prob)

    mean = actor.mean_action(nodes, history)
    assert ((mean > 0) & (mean < 1)).all()

    scored, entropy = actor.evaluate(nodes, history, action)
    assert torch.isfinite(scored) and torch.isfinite(entropy)
    # Re-scoring the same action under the same weights must reproduce the
    # log-prob stored at rollout, or the PPO ratio starts at something other
    # than 1.
    assert scored.item() == pytest.approx(log_prob.item(), abs=1e-5)


def test_beta_parameters_stay_unimodal():
    """alpha, beta >= 1 rules out the U-shaped, bang-bang regime."""
    torch.manual_seed(0)
    actor = InventoryActor(
        node_feature_dim=LOC_DIM + 3, history_dim=1 + 2 * LOOKBACK,
        gin_dims=[16, 16], mlp_dims=[16], embed_dim=8,
    )
    alpha, beta = actor(torch.randn(7, LOC_DIM + 3) * 50, torch.randn(7, 1 + 2 * LOOKBACK) * 50)
    assert (alpha >= 1.0).all() and (beta >= 1.0).all()




def test_tours_always_close_when_rationing_hits_the_capacity_cap():
    """
    Regression: when a period's deliveries are rationed to exactly the vehicle
    capacity, the last stop's float64 delivery can exceed the float32 running
    load by a few parts in 1e6. The serve test and the load mask must agree on
    that node, or it stays selectable while never being servable and the tour
    loops forever.

    Reproduced on this instance at seed 0: delivery 216.5184381508 against a
    remaining load of 216.5184326172, short by 5.5e-6, with the period's total
    rationed to exactly C = 438.
    """
    from environment.irp_env import IRPEnv

    rng = np.random.default_rng(0)
    env = IRPEnv("../data/Instances_highcost_H6/abs3n5.dat", loc_dim=2, lookback_window=3,
                 product_price=None, penalty_factor=None)
    for trial in range(300):
        env.reset()
        terminated = False
        while not terminated:
            delivery = _delivery(rng.uniform(0, 1, env.num_retailers),
                                 env.retailers_current_inventory,
                                 env.current_demand, env.retailer_max_capacity)
            routing_obs, _, _ = env.inventory_action_step(delivery)
            hops = 0
            while True:
                eligible = np.flatnonzero(routing_obs["visited_mask"] == 0)
                assert len(eligible) > 0, f"trial {trial}: every node masked"
                routing_obs, _, critic_obs, terminated, _, _ = env.routing_action_step(
                    int(rng.choice(eligible))
                )
                if critic_obs is not None:
                    break
                hops += 1
                assert hops <= env.num_retailers + 1, (
                    f"trial {trial}: tour did not close after {hops} hops "
                    f"for {env.num_retailers} retailers"
                )
