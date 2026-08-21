import torch
import numpy as np

def build_critic_features(obs):
    """
    Flattens a critic observation dict (see `IRPEnv.critic_observation_space`)
    into a single per-node feature tensor for the critic's GIN encoder.

    Args:
        obs: Critic observation dict. `location` is (num_retailers+1, loc_dim)
            (depot + retailers); `current_inventory`, `holding_cost`,
            `current_demand` are (num_retailers,); `replenishment_history`
            and `historical_demands` are (num_retailers, lookback_window).

        NOTE: `current_inventory`, `holding_cost`, `current_demand`, and the
        history-derived columns below are retailer-only (no depot row), while
        `loc` includes the depot — `np.hstack` requires matching row counts,
        so this raises a shape-mismatch error as written unless the caller
        has already reconciled the depot row.
        NOTE: `curr_demand` is used without a trailing `[:, None]`, unlike
        the other per-node columns, so it isn't reshaped to a column vector
        before stacking.

    Returns:
        Tensor of shape (num_retailers+1, num_features): per-node features
        (location, current inventory, last replenishment, holding cost,
        current demand, last realised demand).
    """
    loc = obs["location"]
    curr_inv = obs["current_inventory"][:, None]
    past_replenishment = obs["replenishment_history"][:, -1:]
    holding_cost = obs["holding_cost"][:, None]
    curr_demand = obs["current_demand"]
    historical_demands = obs["historical_demands"][:, -1:]
    features = np.hstack([loc, curr_inv, past_replenishment, holding_cost, curr_demand, historical_demands])
    return torch.from_numpy(features).float()

def build_inventory_features(obs):
    """
    Flattens an inventory-actor observation dict (see
    `IRPEnv.inventory_observation_space`) into a per-retailer feature tensor.

    Args:
        obs: Inventory observation dict. `location` is (num_retailers,
            loc_dim); `current_inventory` and `replenishment_history` are as
            in `build_critic_features`.

        NOTE: `curr_demand` here reads `historical_demands[:, -1]` (the last
        *realised*, already-consumed demand) rather than `obs["current_demand"]`
        (this period's not-yet-realized demand), even though the latter is
        present in `inventory_observation_space`.

    Returns:
        Tensor of shape (num_retailers, num_features): per-retailer features
        (location, current inventory, last replenishment, last realised
        demand).
    """
    loc = obs["location"]
    curr_inv = obs["current_inventory"][:, None]
    past_replenishment = obs["replenishment_history"][:, -1:]
    curr_demand = obs["historical_demands"][:, -1:]
    features = np.hstack([loc, curr_inv, past_replenishment, curr_demand])
    return torch.from_numpy(features).float()

def build_routing_features(obs):
    """
    Flattens a routing-actor observation dict (see
    `IRPEnv.routing_observation_space`) into a per-node feature tensor.

    Args:
        obs: Routing observation dict. `location` is (num_retailers+1,
            loc_dim) (depot + retailers); `replenishment_amount` is
            (num_retailers,) and is padded with a leading `0.0` for the
            depot row so it aligns with `location`.

    Returns:
        Tensor of shape (num_retailers+1, num_features): per-node features
        (location, replenishment amount to deliver).
    """
    loc = obs["location"]
    replenishment = np.concatenate([[0.0], obs["replenishment_amount"]])
    features = np.hstack([loc, replenishment[:, None]])
    return torch.from_numpy(features).float()
