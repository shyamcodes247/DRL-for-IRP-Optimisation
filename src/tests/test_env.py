import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.irp_env import IRPEnv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
instance_path = DATA_DIR / "Instances_lowcost_H6" / "abs1n5.dat"

env = IRPEnv(str(instance_path), loc_dim=2, lookback_window=3)
inv_obs, critic_obs, info = env.reset()

print("=" * 70)
print(f"INSTANCE: {env.num_retailers} retailers | horizon {env.episode_length} "
      f"| vehicle cap {env.vehicle_capacity}")
print(f"  capacities : {env.retailer_max_capacity}")
print(f"  demand/pd  : {env.demand[:, 0]}")
print(f"  holding    : {env.holding_cost}")
print(f"  depot: inv={env.depot_inventory} rate={env.depot_production_rate}")

for t in range(env.episode_length):
    print("\n" + "=" * 70)
    print(f"PERIOD {t}")
    print(f"  inventory before : {env.retailers_current_inventory}")
    print(f"  depot before     : {env.depot_inventory:.1f}")

    action = env.inventory_action_space.sample()
    print(f"  requested        : {action}")

    routing_obs, r_inv, info = env.inventory_action_step(action)

    print(f"  actually applied : {env.replenishment_amount}")
    print(f"  demand this pd   : {env.current_demand}")
    print(f"  inventory after  : {env.retailers_current_inventory}")
    print(f"  depot after      : {env.depot_inventory:.1f}")
    print(f"  r_inv            : {r_inv:.2f}")

    print(f"  --- routing (start at node {env.vehicle_position}, "
          f"load {env.current_load_capacity[0]:.1f}) ---")

    total_r_vrp, guard = 0.0, 0
    while True:
        mask = routing_obs["visited_mask"]
        eligible = np.flatnonzero(mask == 0)
        print(f"      eligible={eligible.tolist()}  load={routing_obs['current_load_capacity'][0]:.1f}")
        assert len(eligible) > 0, f"DEADLOCK at t={t}"

        a = int(np.random.choice(eligible))
        routing_obs, r_vrp, critic_obs, terminated, truncated, info = env.routing_action_step(a)
        total_r_vrp += r_vrp
        label = "DEPOT" if a == 0 else f"retailer {a}"
        print(f"      -> {label:12s} r_vrp={r_vrp:8.2f}  load now={env.current_load_capacity[0]:.1f}")

        if critic_obs is not None:
            break
        guard += 1
        assert guard < 500, "routing loop did not terminate"

    print(f"  route total r_vrp: {total_r_vrp:.2f}")
    print(f"  replenishment_history:\n{env.replenishment_history}")
    print(f"  historical_demands:\n{env.historical_demands}")

    # invariants
    assert (env.retailers_current_inventory >= 0).all(), "negative inventory"
    assert (env.retailers_current_inventory <= env.retailer_max_capacity).all(), "over capacity"
    assert env.current_load_capacity[0] >= 0, "negative vehicle load"
    assert env.depot_inventory >= 0, "negative depot inventory"

    if terminated:
        print(f"\nEPISODE TERMINATED at t={t}")
        break