import numpy as np
import pandas as pd
import gymnasium as gym
from .data_converter import convert_instance

class IRPEnv(gym.Env):
    """
        Custom Gymnasium environment for the Inventory Routing Problem (IRP)
        under an MTPPO-style (Multi-Task PPO) CTDE architecture.
    
        Follows a two-actor, one-critic design (Lu et al., 2025):
          - Inventory actor: per-node continuous replenishment decisions.
          - Routing actor: sequential discrete node-selection decisions.
          - Critic: centralized value function evaluating the joint state
            of both actors once per timestep, before either decision is made.
    
        Each actor and the critic have their own observation space, since
        they consume different subsets/views of the environment's state.
        The critic has no action_space, since it only estimates value and
        does not select actions.
    """
    def __init__(self, data_file_path, inventory_capacity, loc_dim, lookback_window, max_demand, adjacency_list, min_holding_cost, max_holding_cost):
        """
        Args:
            episode_length: Number of timesteps per episode (planning horizon).
                Also referred to as T / horizon H in the source paper.
            inventory_capacity: Max inventory level a retailer node can hold.
                NOTE: currently applied as a single scalar bound (uniform
                across all retailers). If per-node capacity varies (per the
                benchmark data), this needs to become a shape-(num_retailers,)
                array instead of a scalar.
            num_retailers: Number of retailer nodes in the graph (excludes depot).
            loc_dim: Dimensionality of a node's location feature (e.g. 2 for x,y).
            lookback_window: Number of past periods included in the sliding-window
                history features (replenishment_history, historical_demands).
                NOTE: benchmark instances have very short horizons (e.g. H=3),
                so this may need to be <= episode_length, or reconsidered
                entirely for short-horizon instances.
            max_demand: Upper bound used for demand-related Box spaces.
            loading_capacity: Vehicle's maximum load capacity (Q in the paper).
            adjacency_list: Fixed graph connectivity (neighbor sets per node),
                used by the GIN layers for message passing. Assumes a fixed
                topology for the life of this environment instance.
        """
        params, supplier, retailers = convert_instance(data_file_path).values()
        self.episode_length = params["episode_length"]
        self.num_retailers = params["num_nodes"]
        self.vehicle_capacity = params["vehicle_capacity"]
        self.retailers_initial_inventory = retailers["initial_inventory"].to_numpy()
        self.retailer_min_capacity = retailers["min_capacity"].to_numpy()
        self.retailer_max_capacity = retailers["max_capacity"].to_numpy() # does that mean replenishments are always between 0 and max - min capacity
        self.location = retailers[["x_cord", "y_cord"]].to_numpy()
        self.demand = retailers["demand"].to_numpy()
        self.holding_cost = retailers["holding_cost"].to_numpy() # is holding_cost and demand constant for data here?
        self.depot_location = np.array([[supplier["x_cord"], supplier["y_cord"]]])
        self.depot_initial_inventory = supplier["initial_inventory"]
        self.depot_production_rate = supplier["production_rate"]
        self.depot_holding_cost = supplier["holding_cost"]
        self.loc_dim = loc_dim
        self.lookback_window = lookback_window
        # need to implement this myself (since we have a complete graph)
        self.adjacency_list = adjacency_list
        self.inventory_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers + 1, loc_dim), dtype=np.float32),
                "current_inventory": gym.spaces.Box(low=self.retailer_min_capacity, high=self.retailer_max_capacity, shape=(self.num_retailers,), dtype=np.float32),
                # is current_demand last timestamp's demand?
                "current_demand": gym.spaces.Box(low=0, high=max_demand, shape=(self.num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=min_holding_cost, high=max_holding_cost, shape=(self.num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=inventory_capacity, shape=(self.num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=0, high=max_demand, shape=(self.num_retailers, lookback_window), dtype=np.float32)
            }
        )
        self.inventory_action_space =  gym.spaces.Box(low=0, high=inventory_capacity, shape=(self.num_retailers,), dtype=np.float32)
        self.routing_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers + 1, loc_dim), dtype=np.float32),
                "vehicle_position": gym.spaces.Discrete(self.num_retailers + 1),
                "replenishment_amount": gym.spaces.Box(low=0, high=np.inf, shape=(self.num_retailers, ), dtype=np.float32),
                "current_load_capacity": gym.spaces.Box(low=0, high=self.vehicle_capacity, shape=(1,), dtype=np.float32),
                "visited_mask": gym.spaces.MultiBinary(self.num_retailers + 1)
            }
        )
        self.routing_action_space = gym.spaces.Discrete(self.num_retailers + 1)
        self.critic_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers + 1, loc_dim), dtype=np.float32),

                # Inventory-side info — from inventory_observation_space
                "current_inventory": gym.spaces.Box(low=self.retailer_min_capacity, high=self.retailer_max_capacity, shape=(self.num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=min_holding_cost, high=max_holding_cost, shape=(self.num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=inventory_capacity, shape=(self.num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=0, high=max_demand, shape=(self.num_retailers, lookback_window), dtype=np.float32),

                # Routing-side info — from routing_observation_space
                "current_load_capacity": gym.spaces.Box(low=0, high=self.vehicle_capacity, shape=(1,), dtype=np.float32),
                "visited_mask": gym.spaces.MultiBinary(self.num_retailers + 1),
                "vehicle_position": gym.spaces.Discrete(self.num_retailers + 1),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visited_mask = np.zeros(self.num_nodes, dtype=int)
        self.visited_mask[0] = 1
        self.current_step = 0
        self.retailers_current_inventory = self.retailers_initial_inventory.copy()
        self.replenishment_history = np.zeros(self.num_retailers, self.lookback_window, dtype=np.float32)
        self.historical_demands = np.zeros((self.num_retailers, self.lookback_window), dtype=np.float32)
        self.vehicle_position = 0
        self.current_load_capacity = self.vehicle_capacity
        self.route_log = []

        inventory_obs = {
            "location": self.location,
            "current_inventory": self.retailers_current_inventory,
            "current demand": self.demand,
            "holding_cost": self.holding_cost,
            "replenishment_history": self.replenishment_history,
            "historical_demands": self.demand,
        }

        return inventory_obs

    def inventory_action_step(self, action):
        routing_obs = {
            "location": np.vstack([self.depot_location], [self.location]),
            "vehicle_position": self.vehicle_position,
            "replenishment_amount": action,
            "current_load_capacity": self.vehicle_capacity,
            "visited_mask": self.visited_mask
        }

        r_inv = 0
        for current_inv, unit_holding_cost in zip(self.retailers_current_inventory, self.holding_cost):
            r_inv += current_inv * unit_holding_cost

        r_inv *= -1
        
        return routing_obs, r_inv
    
    def routing_action_step(self, action):
        pass
        
    def render(self):
        pass

    @staticmethod
    def _get_critic_obs(self):
        critic_obs = {
            "location": self.location,
            "current_inventory": self.retailers_inventory,
            "current demand": self.demand,
            "holding_cost": self.holding_cost,
            "replenishment_history": self.replenishment_history,
            "historical_demands": self.demand,
            "current_load_capacity": self.vehicle_capacity,
            "visited_mask": self.visited_mask,
            "vehicle_position": self.vehicle_position
        }

        return critic_obs