import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
import torch
from pathlib import Path
from environment.irp_env import IRPEnv
from agent.critic import Critic
from agent.inventory_actor import InventoryActor
from agent.routing_actor import RoutingActor
from agent.features import (
    build_critic_features,
    build_global_features,
    build_inventory_features,
    build_inventory_history,
    build_routing_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
instance_path = DATA_DIR / "Instances_lowcost_H6" / "abs1n5.dat"

LOC_DIM = 2
LOOKBACK_WINDOW = 3
GIN_DIMS = [64, 128, 128]
MLP_DIMS = [128, 128]
EMBED_DIM = LOC_DIM + 2 * LOOKBACK_WINDOW

env = IRPEnv(str(instance_path), loc_dim=LOC_DIM, lookback_window=LOOKBACK_WINDOW)

critic = Critic(
    node_feature_dim=LOC_DIM + 6,
    global_feature_dim=2,
    hidden_dims=GIN_DIMS,
    mlps_dim=MLP_DIMS,
)
inv_actor = InventoryActor(
    node_feature_dim=LOC_DIM + 3,
    history_dim=1 + 2 * LOOKBACK_WINDOW,
    gin_dims=GIN_DIMS,
    mlp_dims=MLP_DIMS,
    embed_dim=EMBED_DIM,
)
routing_actor = RoutingActor(
    node_feature_dim=LOC_DIM + 1,
    gin_dims=GIN_DIMS,
    mlp_dims=MLP_DIMS,
)

inv_obs, critic_obs, info = env.reset()

v = critic(build_critic_features(critic_obs), build_global_features(critic_obs))
mu, sigma = inv_actor(build_inventory_features(inv_obs),
                      build_inventory_history(inv_obs))
print(v.shape, mu.shape, sigma.shape)     # (1,), (n,), (n,)

action, logp = inv_actor.act(build_inventory_features(inv_obs), build_inventory_history(inv_obs))
routing_obs, r_inv, info = env.inventory_action_step(action.numpy())
mask = torch.from_numpy(routing_obs["visited_mask"])
node, logp_r = routing_actor.act(build_routing_features(routing_obs), mask)
print(node, logp_r)
