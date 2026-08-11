import numpy as np
import pandas as pd
import gymnasium as gym

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
    def __init__(self, data_file_path, episode_length, inventory_capacity, num_retailers, loc_dim, lookback_window, max_demand, loading_capacity, adjacency_list, min_holding_cost, max_holding_cost, supplier_initial_inventory, supplier_production_rate, supplier_holding_cost):
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
        data_file_df = pd.read_csv(data_file_path, '\t')
        self.episode_length = episode_length
        self.current_step = 0
        self.inventory_capacity = inventory_capacity
        self.num_retailers = num_retailers
        self.loc_dim = loc_dim
        self.lookback_window = lookback_window
        self.max_demand = max_demand
        self.loading_capacity = loading_capacity
        self.adjacency_list = adjacency_list
        self.supplier_initial_inventory = supplier_initial_inventory
        self.supplier_production_rate = supplier_production_rate
        self.supplier_holding_cost = supplier_holding_cost
        self.inventory_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_retailers + 1, loc_dim), dtype=np.float32),
                "current_inventory": gym.spaces.Box(low=0, high=inventory_capacity, shape=(num_retailers,), dtype=np.float32),
                "current_demand": gym.spaces.Box(low=0, high=max_demand, shape=(num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=min_holding_cost, high=max_holding_cost, shape=(num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=inventory_capacity, shape=(num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=0, high=max_demand, shape=(num_retailers, lookback_window), dtype=np.float32)
            }
        )
        self.inventory_action_space =  gym.spaces.Box(low=0, high=inventory_capacity, shape=(num_retailers,), dtype=np.float32)
        self.routing_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_retailers + 1, loc_dim), dtype=np.float32),
                "vehicle_position": gym.spaces.Discrete(num_retailers + 1),
                "replenishment_amount": gym.spaces.Box(low=0, high=np.inf, shape=(num_retailers, ), dtype=np.float32),
                "current_load_capacity": gym.spaces.Box(low=0, high=loading_capacity, shape=(1,), dtype=np.float32),
                "visited_mask": gym.spaces.MultiBinary(num_retailers + 1)
            }
        )
        self.routing_action_space = gym.spaces.Discrete(num_retailers + 1)
        self.critic_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_retailers + 1, loc_dim), dtype=np.float32),

                # Inventory-side info — from inventory_observation_space
                "current_inventory": gym.spaces.Box(low=0, high=inventory_capacity, shape=(num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=min_holding_cost, high=max_holding_cost, shape=(num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=inventory_capacity, shape=(num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=0, high=max_demand, shape=(num_retailers, lookback_window), dtype=np.float32),

                # Routing-side info — from routing_observation_space
                "current_load_capacity": gym.spaces.Box(low=0, high=loading_capacity, shape=(1,), dtype=np.float32),
                "visited_mask": gym.spaces.MultiBinary(num_retailers + 1),
                "vehicle_position": gym.spaces.Discrete(num_retailers + 1),
            }
        )

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.current_inventory = self.initial_inventory.copy()

    def inventory_action_step(self, action):
        pass
    
    def routing_action_step(self, action):
        pass
        
    def render(self):
        pass