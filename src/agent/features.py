from typing import Dict

import torch
import numpy as np
import numpy.typing as npt

def build_critic_features(obs: Dict[str, npt.NDArray]) -> torch.Tensor:
    """
    Flattens a critic observation dict (see `IRPEnv.critic_observation_space`)
    into a single per-node feature tensor for the critic's GIN encoder.

    Production rate is a fixed global scalar, so it is handled separately by
    `build_global_features` rather than folded in here. Depot inventory,
    however, is included as a node feature: it varies over the episode like
    the retailer-only columns do, just on the depot row instead of theirs.
    Each retailer-only column (current inventory, holding cost, current
    demand, last replenishment, last realised demand) is zero-padded on the
    depot row, and depot inventory is zero-padded on the retailer rows —
    this keeps every node's feature vector the same width, as
    `nn.Sequential`/GIN require.

    Args:
        obs: Critic observation dict. `location` is (num_retailers+1, loc_dim)
            (depot + retailers, depot first); `current_inventory`,
            `holding_cost`, `current_demand` are (num_retailers,);
            `replenishment_history` and `historical_demands` are
            (num_retailers, lookback_window); `depot_inventory` is (1,).

    Returns:
        Tensor of shape (num_retailers+1, num_features): per-node features
        (location, current inventory, last replenishment, holding cost,
        current demand, last realised demand, depot inventory).
    """
    num_retailers = obs["location"].shape[0] - 1

    def pad_depot_row(retailer_col: npt.NDArray) -> npt.NDArray:
        return np.vstack([np.zeros((1, retailer_col.shape[1])), retailer_col])

    def pad_retailer_rows(depot_col: npt.NDArray) -> npt.NDArray:
        return np.vstack([depot_col, np.zeros((num_retailers, depot_col.shape[1]))])

    loc = obs["location"]
    curr_inv = pad_depot_row(obs["current_inventory"][:, None])
    past_replenishment = pad_depot_row(obs["replenishment_history"][:, -1:])
    holding_cost = pad_depot_row(obs["holding_cost"][:, None])
    curr_demand = pad_depot_row(obs["current_demand"][:, None])
    historical_demands = pad_depot_row(obs["historical_demands"][:, -1:])
    depot_inventory = pad_retailer_rows(obs["depot_inventory"][:, None])
    features = np.hstack([
        loc, curr_inv, past_replenishment, holding_cost, curr_demand, historical_demands,
        depot_inventory,
    ])
    return torch.from_numpy(features).float()


def build_global_features(obs: Dict[str, npt.NDArray]) -> torch.Tensor:
    """
    Flattens the critic observation's global, non-per-node scalars into a
    single feature vector — consumed by the critic after GIN pooling, rather
    than as padded node columns (see `build_critic_features`).

    Args:
        obs: Critic observation dict. `production_rate` and
            `normalised_current_step` are each (1,).

    Returns:
        Tensor of shape (num_global_features,): (production rate, normalised
        current step).
    """
    production_rate = obs["production_rate"]
    normalised_current_step = obs["normalised_current_step"]
    features = np.concatenate([production_rate, normalised_current_step])
    return torch.from_numpy(features).float()

def build_inventory_features(obs: Dict[str, npt.NDArray]) -> torch.Tensor:
    """
    Flattens an inventory-actor observation dict (see
    `IRPEnv.inventory_observation_space`) into a per-retailer feature tensor.

    Args:
        obs: Inventory observation dict. `location` is (num_retailers,
            loc_dim); `current_inventory` and `replenishment_history` are as
            in `build_critic_features`.

    Returns:
        Tensor of shape (num_retailers, num_features): per-retailer features
        (location, current inventory, last replenishment, last realised
        demand).
    """
    loc = obs["location"]
    curr_inv = obs["current_inventory"][:, None]
    past_replenishment = obs["replenishment_history"][:, -1:]
    curr_demand = obs["current_demand"][:, None]
    features = np.hstack([loc, curr_inv, past_replenishment, curr_demand])
    return torch.from_numpy(features).float()

def build_routing_features(obs: Dict[str, npt.NDArray]) -> torch.Tensor:
    """
    Flattens a routing-actor observation dict (see
    `IRPEnv.routing_observation_space`) into a per-node feature tensor.

    Includes each node's displacement from the vehicle's *current* node, so
    the policy can tell a nearby stop from a distant one. Lu et al. (2025)
    declare the vehicle's route/position in the routing state (Eq. 27) but
    their Eq. (30) initialises the GIN with location and replenishment
    quantity only, and Eq. (32)'s decoder takes just the pooled encoder
    output and the mask — so a literal implementation cannot condition on
    where the vehicle is, and can only ever emit a fixed priority order over
    nodes. Measured on a trained checkpoint of that literal version: the
    logits were bit-identical whether the vehicle sat on a node or 309
    distance units away, the same visit order was emitted every period, and
    the resulting tours were longer than ~60-78% of random orderings of the
    same node set (2.1x longer than 2-opt at n=10, 3.3x at n=30).

    NOTE: the paper's state also carries the vehicle's current load, which is
    deliberately not included here: `IRPEnv` caps a period's total deliveries
    at one vehicle load and has no mid-tour reload, so remaining load is a
    monotone function of what has already been visited and can never make a
    node infeasible. Add it if that constraint is ever relaxed.

    Args:
        obs: Routing observation dict. `location` is (num_retailers+1,
            loc_dim) (depot + retailers); `replenishment_amount` is
            (num_retailers,) and is padded with a leading `0.0` for the
            depot row so it aligns with `location`; `vehicle_position` is
            the index of the node the vehicle currently sits on.

    Returns:
        Tensor of shape (num_retailers+1, 2*loc_dim + 2): per-node features
        (location, displacement from the vehicle's current node,
        replenishment amount to deliver, whether this is the current node).
        The leading 2*loc_dim columns are both coordinate-scaled — see
        `MTPPO._normalize_location`'s `num_coord_cols`.
    """
    loc = obs["location"]
    position = int(obs["vehicle_position"])
    displacement = loc - loc[position]
    replenishment = np.concatenate([[0.0], obs["replenishment_amount"]])
    is_current = np.zeros((loc.shape[0], 1))
    is_current[position] = 1.0
    features = np.hstack([loc, displacement, replenishment[:, None], is_current])
    return torch.from_numpy(features).float()

def build_inventory_history(obs: Dict[str, npt.NDArray]) -> torch.Tensor:
    inv = obs["current_inventory"][:, None]
    rep = obs["replenishment_history"]
    dem = obs["historical_demands"]
    
    return torch.from_numpy(np.hstack([inv, rep, dem])).float()