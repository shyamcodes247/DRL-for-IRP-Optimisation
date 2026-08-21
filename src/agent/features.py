import torch
import numpy as np

def build_critic_features(obs):
    loc = obs["location"]
    curr_inv = obs["current_inventory"][:, None]
    past_replenishment = obs["replenishment_history"][:, -1:]
    holding_cost = obs["holding_cost"][:, None]
    curr_demand = obs["current_demand"]
    historical_demands = obs["historical_demands"][:, -1:]
    features = np.hstack([loc, curr_inv, past_replenishment, holding_cost, curr_demand, historical_demands])
    return torch.from_numpy(features).float()

def build_inventory_features(obs):
    loc = obs["location"]
    curr_inv = obs["current_inventory"][:, None]
    past_replenishment = obs["replenishment_history"][:, -1:]
    curr_demand = obs["historical_demands"][:, -1:]
    features = np.hstack([loc, curr_inv, past_replenishment, curr_demand])
    return torch.from_numpy(features).float()

def build_routing_features(obs):
    loc = obs["location"]
    replenishment = np.concatenate([[0.0], obs["replenishment_amount"]])
    features = np.hstack([loc, replenishment[:, None]])
    return torch.from_numpy(features).float()
