import numpy as np
import numpy.typing as npt
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
    def __init__(self, data_file_path, loc_dim, lookback_window, product_price=None, penalty_factor=None, delivery_cost=1):
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
        self.num_retailers = params["num_nodes"] - 1 # the data file includes supplier as node
        self.vehicle_capacity = params["vehicle_capacity"]
        self.product_price = product_price
        self.penalty_factor = penalty_factor
        self.retailers_initial_inventory = retailers["initial_inventory"].to_numpy()
        self.retailer_min_capacity = retailers["min_capacity"].to_numpy()
        self.retailer_max_capacity = retailers["max_capacity"].to_numpy()
        self.location = retailers[["x_cord", "y_cord"]].to_numpy()
        # need to implement a way to tell whether there is single or varying demand given
        self.demand = retailers["demand"].to_numpy()
        self.demand = np.tile(self.demand[:, None], (1, self.episode_length))
        self.holding_cost = retailers["holding_cost"].to_numpy()
        self.depot_location = np.array([[supplier["x_cord"], supplier["y_cord"]]])
        self.depot_initial_inventory = supplier["initial_inventory"]
        self.depot_production_rate = supplier["production_rate"]
        self.depot_holding_cost = supplier["holding_cost"]
        self.loc_dim = loc_dim
        self.lookback_window = lookback_window
        # need to implement this myself (since we have a complete graph)
        self.adjacency_list = self._create_adjacency_list(self.num_retailers + 1)
        self.delivery_cost = delivery_cost
        max_demand = np.max(self.demand, axis=1)
        min_demand = np.min(self.demand, axis=1)
        
        self.inventory_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers, loc_dim), dtype=np.float32),
                "current_inventory": gym.spaces.Box(low=self.retailer_min_capacity, high=self.retailer_max_capacity, shape=(self.num_retailers,), dtype=np.float32),
                "current_demand": gym.spaces.Box(low=min_demand, high=max_demand, shape=(self.num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=self.holding_cost, high=self.holding_cost, shape=(self.num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=np.subtract(self.retailer_max_capacity, self.retailer_min_capacity), shape=(self.num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=min_demand, high=max_demand, shape=(self.num_retailers, lookback_window), dtype=np.float32)
            }
        )
        self.inventory_action_space =  gym.spaces.Box(low=0, high=np.subtract(self.retailer_max_capacity, self.retailer_min_capacity), shape=(self.num_retailers,), dtype=np.float32)
        self.routing_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers + 1, loc_dim), dtype=np.float32),
                "vehicle_position": gym.spaces.Discrete(self.num_retailers + 1),
                "replenishment_amount": gym.spaces.Box(low=0, high=np.subtract(self.retailer_max_capacity, self.retailer_min_capacity), shape=(self.num_retailers, ), dtype=np.float32),
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
                "current_demand": gym.spaces.Box(low=min_demand, high=max_demand, shape=(self.num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=self.holding_cost, high=self.holding_cost, shape=(self.num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=np.subtract(self.retailer_max_capacity, self.retailer_min_capacity), shape=(self.num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=min_demand, high=max_demand, shape=(self.num_retailers, lookback_window), dtype=np.float32),

                # Routing-side info — from routing_observation_space
                "current_load_capacity": gym.spaces.Box(low=0, high=self.vehicle_capacity, shape=(1,), dtype=np.float32),
                "visited_mask": gym.spaces.MultiBinary(self.num_retailers + 1),
                "vehicle_position": gym.spaces.Discrete(self.num_retailers + 1),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visited_mask = np.zeros(self.num_retailers + 1, dtype=int)
        self.current_step = 0
        self.current_demand = np.zeros(self.num_retailers, dtype=np.float32)
        self.retailers_current_inventory = self.retailers_initial_inventory.copy()
        self.replenishment_amount = np.zeros(self.num_retailers, dtype=np.float32)
        self.replenishment_history = np.zeros((self.num_retailers, self.lookback_window), dtype=np.float32)
        self.historical_demands = np.zeros((self.num_retailers, self.lookback_window), dtype=np.float32)
        self.vehicle_position = 0
        self.current_load_capacity = np.array([self.vehicle_capacity], dtype=np.float32)
        self.depot_inventory = self.depot_initial_inventory
        self.route_log = []

        inventory_obs = {
            "location": self.location,
            "current_inventory": self.retailers_current_inventory,
            "current_demand": self.current_demand,
            "holding_cost": self.holding_cost,
            "replenishment_history": self.replenishment_history,
            "historical_demands": self.historical_demands,
        }

        critic_obs = {
            "location": np.vstack([self.depot_location, self.location]),
            "current_inventory": self.retailers_current_inventory,
            "current_demand": self.current_demand,
            "holding_cost": self.holding_cost,
            "replenishment_history": self.replenishment_history,
            "historical_demands": self.historical_demands,
            "current_load_capacity": self.current_load_capacity,
            "visited_mask": self.visited_mask,
            "vehicle_position": self.vehicle_position
        }

        info = {}

        return inventory_obs, critic_obs, info

    def inventory_action_step(self, action):
        self.current_demand = self.demand[:, self.current_step]

        # Updates the depot's inventory levels based on delivery amounts
        self.depot_inventory += self.depot_production_rate
        # checks if amount to be delivered exceeds the inventory amount of depot's inventory
        if np.sum(action) > self.depot_inventory:
            scale = self.depot_inventory / np.sum(action) 
            action = action * scale
        
        # Ensures that any action that results in a break of the max_capacity of retailer is capped
        max_delivery_allowed =  np.minimum( 
            self.retailer_max_capacity - self.retailers_current_inventory,
            self.vehicle_capacity
        )
        action = np.clip(action, 0, np.maximum(max_delivery_allowed, 0))

        self.depot_inventory -= np.sum(action)
        self.replenishment_amount = action
        routing_obs = {
            "location": np.vstack([self.depot_location, self.location]),
            "vehicle_position": self.vehicle_position,
            "replenishment_amount": self.replenishment_amount,
            "current_load_capacity": self.current_load_capacity,
            "visited_mask": self.visited_mask
        }
            
        self.retailers_current_inventory += action
        sales_loss = self.current_demand - self.retailers_current_inventory
        sales_loss = np.maximum(sales_loss, 0)
        self.retailers_current_inventory -= self.current_demand
        self.retailers_current_inventory += sales_loss

        r_inv = self.depot_inventory * self.depot_holding_cost
        for current_inv, unit_holding_cost in zip(self.retailers_current_inventory, self.holding_cost):
            r_inv += current_inv * unit_holding_cost
        
        for sales_loss_cost in sales_loss:
            if self.product_price is not None and self.penalty_factor is not None:
                r_inv += sales_loss_cost * self.product_price * self.penalty_factor
    
        r_inv *= -1
        
        return routing_obs, r_inv
    
    def routing_action_step(self, action: int):
        node_1 = self.depot_location[0] if self.vehicle_position == 0 else self.location[self.vehicle_position - 1]
        node_2 = self.depot_location[0] if action == 0 else self.location[action - 1]
        distance_cost = self._get_distance(node_1=node_1, node_2=node_2)
        
        infeasible = self.replenishment_amount > self.current_load_capacity[0]
        load_mask = np.zeros(self.num_retailers + 1, dtype=int)
        load_mask[1:] = infeasible
        effective_mask = np.maximum(self.visited_mask, load_mask)
        
        if action != 0 and self.replenishment_amount[action - 1] <= self.current_load_capacity[0]:
            self.visited_mask[action] = 1
            effective_mask[action] = 1
            self.vehicle_position = action
            self.current_load_capacity -= np.array([self.replenishment_amount[action - 1]], dtype=np.float32)
        elif action == 0:
            self.current_load_capacity = np.array([self.vehicle_capacity], dtype=np.float32)
            
        routing_obs = {
            "location": np.vstack([self.depot_location, self.location]),
            "vehicle_position": self.vehicle_position,
            "replenishment_amount": self.replenishment_amount,
            "current_load_capacity": self.current_load_capacity,
            "visited_mask": effective_mask
        }

        r_vrp = -distance_cost * self.delivery_cost

        if np.all(self.visited_mask == 1):
            self.current_step += 1
            
            # Update the historical data arrays with replenishment amounts and demands
            self.historical_demands = self._update_history_window(self.historical_demands, self.current_demand)
            self.replenishment_history = self._update_history_window(self.replenishment_history, self.replenishment_amount)
            
            self.visited_mask = np.zeros(self.num_retailers + 1, dtype=int)
            self.current_load_capacity = np.array([self.vehicle_capacity], dtype=np.float32)
            terminated = self.current_step >= self.episode_length
            truncated = False
            critic_obs = {
                "location": np.vstack([self.depot_location, self.location]),
                "current_inventory": self.retailers_current_inventory,
                "current_demand": self.current_demand,
                "holding_cost": self.holding_cost,
                "replenishment_history": self.replenishment_history,
                "historical_demands": self.historical_demands,
                "current_load_capacity": self.current_load_capacity,
                "visited_mask": self.visited_mask,
                "vehicle_position": self.vehicle_position
            }

            info = {}

            return routing_obs, r_vrp, critic_obs, terminated, truncated, info
        else:
            return routing_obs, r_vrp, None, False, False, {}
        
        
    def render(self):
        pass

    # Does critic network rely on the initial state of environment at time t?

    # Returns distance between two nodes
    def _get_distance(self, node_1: npt.NDArray[np.float32], node_2: npt.NDArray[np.float32]):
        return np.linalg.norm(
            node_1 - node_2, ord=2
        )
        
    # Updates history arrays based on movement in time
    def _update_history_window(self, history_arr, current_value):
        history_arr = np.roll(history_arr, shift=-1, axis=1)
        history_arr[:, -1] = current_value
        return history_arr

    def _create_adjacency_list(self, num_nodes):
        return {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}
        