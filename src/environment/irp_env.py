from typing import Any, Dict, List, Optional, Tuple

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

        Node indexing convention
        ------------------------
        Node 0 is always the depot/supplier; nodes 1..num_retailers are the
        retailers. Arrays sized (num_retailers,) are retailer-only and are
        indexed by `node_index - 1`; arrays sized (num_retailers + 1,)
        (e.g. `visited_mask`, `location` in the routing/critic observations)
        include the depot at index 0.

        Interaction loop
        ----------------
        This environment does NOT implement the single-`step()` Gymnasium API,
        because one timestep contains one inventory decision followed by a
        variable-length sequence of routing decisions. Drive it as::

            inv_obs, critic_obs, info = env.reset()
            while not terminated:
                # critic_obs is the joint state, evaluated before either actor acts
                route_obs, r_inv, inv_info = env.inventory_action_step(inv_action)
                while critic_obs is None:
                    route_obs, r_vrp, critic_obs, terminated, truncated, info = \\
                        env.routing_action_step(route_action)

        `routing_action_step` returns `critic_obs = None` while the vehicle
        is still mid-tour, and returns a non-None `critic_obs` on the step
        that closes the tour (all retailers visited) and advances the clock.

        NOTE: `reset()` returns the inventory observation and critic observation
        but not a routing observation — the routing actor only becomes active
        after `inventory_action_step` has produced replenishment amounts.
    """
    def __init__(
        self,
        data_file_path: str,
        loc_dim: int,
        lookback_window: int,
        product_price: Optional[float] = None,
        penalty_factor: Optional[float] = None,
        delivery_cost: float = 1,
    ) -> None:
        """
        Args:
            data_file_path: Path to a benchmark instance file, parsed by
                `convert_instance`. Supplies the planning horizon, node count,
                vehicle capacity, supplier row and retailer table.
            loc_dim: Dimensionality of a node's location feature (2 for x,y).
            lookback_window: Number of past periods included in the sliding-window
                history features (`replenishment_history`, `historical_demands`).
                NOTE: benchmark instances have very short horizons (e.g. H=3),
                so this may need to be <= episode_length, or reconsidered
                entirely for short-horizon instances.
            product_price: Unit selling price of the product. Used together with
                `penalty_factor` to price unmet demand. If either this or
                `penalty_factor` is None, lost sales are not charged at all and
                the inventory reward reduces to holding cost only.
            penalty_factor: Multiplier applied to `product_price` for each unit of
                unmet demand (the stockout penalty is
                `lost_units * product_price * penalty_factor`).
            delivery_cost: Cost per unit of travel distance. Scales the routing
                reward; defaults to 1 so that the routing reward is the negated
                raw tour distance.

        Attributes set from the instance file:
            episode_length: Number of timesteps per episode (planning horizon).
                Also referred to as T / horizon H in the source paper.
            num_retailers: Number of retailer nodes (instance node count minus
                the supplier, which the data file includes as a node).
            vehicle_capacity: Vehicle's maximum load capacity (Q in the paper).
            retailers_initial_inventory / retailer_min_capacity /
            retailer_max_capacity / holding_cost: Per-retailer arrays of shape
                (num_retailers,).
            location: Retailer coordinates, shape (num_retailers, loc_dim).
            depot_location: Depot coordinates, shape (1, loc_dim), kept 2-D so it
                can be `vstack`ed on top of `location` to build the full node list.
            demand: Per-retailer demand over the horizon, shape
                (num_retailers, episode_length).
            adjacency_list: Fixed graph connectivity (neighbor sets per node),
                used by the GIN layers for message passing. Assumes a fixed
                topology (currently a complete graph) for the life of this
                environment instance.
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
        # Benchmark instances give one stationary demand value per retailer, so it is
        # broadcast across the horizon to give a (num_retailers, episode_length) matrix.
        # Time-varying instances would populate this matrix directly instead.
        self.demand = np.tile(self.demand[:, None], (1, self.episode_length))
        self.holding_cost = retailers["holding_cost"].to_numpy()
        self.depot_location = np.array([[supplier["x_cord"], supplier["y_cord"]]])
        self.depot_initial_inventory = supplier["initial_inventory"]
        self.depot_production_rate = supplier["production_rate"]
        self.depot_holding_cost = supplier["holding_cost"]
        self.loc_dim = loc_dim
        self.lookback_window = lookback_window
        self.adjacency_list = self._create_adjacency_list(self.num_retailers + 1)
        self.delivery_cost = delivery_cost

        # Per-node bounds for the Box spaces below. Demand bounds are taken over the
        # whole horizon so the spaces stay valid at every timestep; the "historical"
        # variants are the same bounds tiled across the lookback window.
        max_demand = np.max(self.demand, axis=1)
        min_demand = np.min(self.demand, axis=1)

        max_historical_demand = np.tile(np.max(self.demand, axis=1).reshape(-1, 1), (1, lookback_window))
        min_historical_demand = np.tile(np.min(self.demand, axis=1).reshape(-1, 1), (1, lookback_window))

        # Largest useful delivery to a retailer is the span between its min and max
        # capacity, i.e. topping an empty-but-legal node up to full.
        max_replenishment = np.subtract(self.retailer_max_capacity, self.retailer_min_capacity)
        max_historical_replenishment = np.tile(np.subtract(self.retailer_max_capacity, self.retailer_min_capacity).reshape(-1, 1), (1, lookback_window))

        # Inventory actor's view: retailer nodes only (no depot), no routing state.
        # `holding_cost` is a degenerate Box (low == high) since it is a constant
        # per-node feature rather than a varying observation.
        self.inventory_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers, loc_dim), dtype=np.float32),
                "current_inventory": gym.spaces.Box(low=self.retailer_min_capacity, high=self.retailer_max_capacity, shape=(self.num_retailers,), dtype=np.float32),
                "current_demand": gym.spaces.Box(low=min_demand, high=max_demand, shape=(self.num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=self.holding_cost, high=self.holding_cost, shape=(self.num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=max_historical_replenishment, shape=(self.num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=min_historical_demand, high=max_historical_demand, shape=(self.num_retailers, lookback_window), dtype=np.float32)
            }
        )
        # One continuous replenishment quantity per retailer, decided in a single shot
        # at the start of the timestep.
        self.inventory_action_space =  gym.spaces.Box(low=0, high=max_replenishment, shape=(self.num_retailers,), dtype=np.float32)

        # Routing actor's view: depot + retailers, and the routing state that evolves
        # within a timestep. `replenishment_amount` is the inventory actor's (clipped)
        # decision, which the routing actor treats as fixed per-node demand to serve.
        self.routing_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers + 1, loc_dim), dtype=np.float32),
                "vehicle_position": gym.spaces.Discrete(self.num_retailers + 1),
                "replenishment_amount": gym.spaces.Box(low=0, high=max_replenishment, shape=(self.num_retailers, ), dtype=np.float32),
                "current_load_capacity": gym.spaces.Box(low=0, high=self.vehicle_capacity, shape=(1,), dtype=np.float32),
                "visited_mask": gym.spaces.MultiBinary(self.num_retailers + 1)
            }
        )
        # Index of the next node to visit; 0 is the depot (return-to-depot / reload).
        self.routing_action_space = gym.spaces.Discrete(self.num_retailers + 1)

        # Centralised critic's view: the union of both actors' observations, so it can
        # value the joint state once per timestep before either actor acts (CTDE).
        self.critic_observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_retailers + 1, loc_dim), dtype=np.float32),

                # Inventory-side info — from inventory_observation_space
                "current_inventory": gym.spaces.Box(low=self.retailer_min_capacity, high=self.retailer_max_capacity, shape=(self.num_retailers,), dtype=np.float32),
                "current_demand": gym.spaces.Box(low=min_demand, high=max_demand, shape=(self.num_retailers,), dtype=np.float32),
                "holding_cost": gym.spaces.Box(low=self.holding_cost, high=self.holding_cost, shape=(self.num_retailers,), dtype=np.float32),
                "replenishment_history": gym.spaces.Box(low=0, high=max_historical_replenishment, shape=(self.num_retailers, lookback_window), dtype=np.float32),
                "historical_demands": gym.spaces.Box(low=min_historical_demand, high=max_historical_demand, shape=(self.num_retailers, lookback_window), dtype=np.float32),

                # Routing-side info — from routing_observation_space
                "current_load_capacity": gym.spaces.Box(low=0, high=self.vehicle_capacity, shape=(1,), dtype=np.float32),
                "vehicle_position": gym.spaces.Discrete(self.num_retailers + 1),
            }
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, npt.NDArray], Dict[str, npt.NDArray], Dict[str, Any]]:
        """
        Start a new episode: restore inventories to their instance values, park the
        vehicle at the depot with a full load, and clear the history windows.

        Returns:
            inventory_obs: Observation for the inventory actor's first decision.
            critic_obs: Joint-state observation for the critic at t=0.
            info: Empty dict, kept for Gymnasium-style call compatibility.

        NOTE: this returns a 3-tuple rather than Gymnasium's (obs, info) 2-tuple,
        because two networks need to be fed before the first action is taken.
        """
        super().reset(seed=seed)
        # Depot starts marked as visited so the routing actor is not rewarded for
        # "arriving" at the node it is already parked on.
        self.visited_mask = np.zeros(self.num_retailers + 1, dtype=int)
        self.visited_mask[0] = 1
        self.current_step = 0
        self.current_demand = self.demand[:, self.current_step]
        self.retailers_current_inventory = self.retailers_initial_inventory.copy()
        self.replenishment_amount = np.zeros(self.num_retailers, dtype=np.float32)
        self.replenishment_history = np.zeros((self.num_retailers, self.lookback_window), dtype=np.float32)
        self.historical_demands = np.zeros((self.num_retailers, self.lookback_window), dtype=np.float32)
        self.vehicle_position = 0
        self.current_load_capacity = np.array([self.vehicle_capacity], dtype=np.float32)
        self.depot_inventory = self.depot_initial_inventory
        self.route_log = []  # reserved for trajectory logging / rendering; not yet populated

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
            "vehicle_position": self.vehicle_position
        }

        info = {}

        return inventory_obs, critic_obs, info

    def inventory_action_step(
        self, action: npt.NDArray[np.float32]
    ) -> Tuple[Dict[str, npt.NDArray], float, Dict[str, Any]]:
        """
        Apply the inventory actor's replenishment decision for the current timestep.

        The requested quantities are made feasible before being committed, in order:
          1. production arrives at the depot,
          2. the whole request is scaled down proportionally if it exceeds depot stock,
          3. each node's quantity is capped by its remaining headroom and by the
             vehicle capacity,
        after which stock moves depot -> retailers, demand is realised, and the
        inventory-side cost is charged.

        Args:
            action: Requested replenishment per retailer, shape (num_retailers,).
                Treated as a request, not a guarantee — the returned
                `routing_obs["replenishment_amount"]` holds the quantities that
                were actually committed and that the routing actor must deliver.

        Returns:
            routing_obs: Observation handed to the routing actor for this timestep.
            r_inv: Inventory reward — the negated sum of depot holding, retailer
                holding and lost-sales penalty.
            info: Per-step inventory diagnostics (cost breakdown, lost sales units,
                stockout count).

        NOTE: `routing_obs` is built *before* demand is subtracted, so the routing
        actor sees the post-delivery/pre-consumption state.
        """

        # Updates the depot's inventory levels based on delivery amounts
        self.depot_inventory += self.depot_production_rate
        # checks if amount to be delivered exceeds the inventory amount of depot's inventory
        # Scaled proportionally rather than truncated, so the actor's relative
        # allocation across retailers is preserved when stock is short.
        if np.sum(action) > self.depot_inventory:
            scale = self.depot_inventory / np.sum(action)
            action = action * scale

        # Ensures that any action that results in a break of the max_capacity of retailer is capped
        # Also bounded by vehicle capacity, since a single node's delivery can never
        # exceed one full vehicle load.
        max_delivery_allowed =  np.minimum(
            self.retailer_max_capacity - self.retailers_current_inventory,
            self.vehicle_capacity
        )
        # `np.maximum(..., 0)` guards against a negative upper bound when a node is
        # already at or above its max capacity.
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
            
        # Deliver, then realise demand. Unmet demand is lost (not backordered): adding
        # `sales_loss` back after subtracting demand floors inventory at zero, and the
        # shortfall is charged below instead of being carried into the next period.
        self.retailers_current_inventory += action
        sales_loss = self.current_demand - self.retailers_current_inventory
        sales_loss = np.maximum(sales_loss, 0)
        self.retailers_current_inventory -= self.current_demand
        self.retailers_current_inventory += sales_loss

        # Holding cost is charged on end-of-period stock at both echelons.
        depot_holding = self.depot_inventory * self.depot_holding_cost
        retailer_holding = 0
        for current_inv, unit_holding_cost in zip(self.retailers_current_inventory, self.holding_cost):
            retailer_holding += current_inv * unit_holding_cost

        # Stockouts are only priced when both pricing parameters were supplied;
        # otherwise lost sales are recorded in `info` but cost nothing.
        lost_sales_cost = 0
        for sales_lost in sales_loss:
            if self.product_price is not None and self.penalty_factor is not None:
                lost_sales_cost += sales_lost * self.product_price * self.penalty_factor

        r_inv = - (depot_holding + retailer_holding + lost_sales_cost)

        info = {
            "retailer_holding_cost": retailer_holding,
            "depot_holding_cost": depot_holding,
            "lost_sales_units": float(np.sum(sales_loss)),
            "stockout_count": int(np.sum(sales_loss > 0)),
        }

        
        return routing_obs, r_inv, info
    
    def routing_action_step(
        self, action: int
    ) -> Tuple[Dict[str, npt.NDArray], float, Optional[Dict[str, npt.NDArray]], bool, bool, Dict[str, Any]]:
        """
        Move the vehicle to one node. Called repeatedly within a timestep until
        every retailer has been served, which closes the tour and advances the clock.

        Args:
            action: Index of the node to move to. 0 is the depot, which reloads the
                vehicle to full capacity; 1..num_retailers are retailers, which are
                served with their full `replenishment_amount`.

        Returns:
            routing_obs: Next routing observation. Its `visited_mask` is the
                *effective* mask (already-visited nodes plus nodes the current load
                cannot serve), so the policy can mask its logits directly.
            r_vrp: Routing reward — negated travel distance scaled by `delivery_cost`.
                Zero for an infeasible move (see below).
            critic_obs: Joint state for the next timestep, or None if the tour is
                still in progress. A non-None value signals the timestep boundary.
            terminated: True once `current_step` reaches the planning horizon.
            truncated: Always False; there is no time-limit truncation.
            info: Empty dict.

        Split deliveries are not modelled: a retailer is served in one visit, so a
        node whose requested amount exceeds the remaining load is unreachable until
        the vehicle returns to the depot to reload.
        """
        # Resolve both endpoints to coordinates. Retailer arrays are offset by one
        # because index 0 is the depot.
        node_1 = self.depot_location[0] if self.vehicle_position == 0 else self.location[self.vehicle_position - 1]
        node_2 = self.depot_location[0] if action == 0 else self.location[action - 1]
        distance_cost = self._get_distance(node_1=node_1, node_2=node_2)

        if action != 0 and self.replenishment_amount[action - 1] <= self.current_load_capacity[0]:
            # Feasible retailer visit: mark served and unload.
            self.visited_mask[action] = 1
            self.current_load_capacity -= np.array([self.replenishment_amount[action - 1]], dtype=np.float32)
            self.vehicle_position = action
        elif action == 0:
            # Returning to the depot reloads the vehicle to full capacity.
            self.current_load_capacity = np.array([self.vehicle_capacity], dtype=np.float32)
            self.vehicle_position = action

        # Forces agent to reconsider its action by returning zero reward and masks node out to ensure it is not chosen again
        # The vehicle does not move in this case, so charging travel would be wrong;
        # the node is excluded by `load_mask` below until a reload makes it feasible.
        if action != 0 and self.replenishment_amount[action - 1] > self.current_load_capacity[0]:
            distance_cost = 0

        # The depot is only "visited" while the vehicle sits on it; leaving it re-opens
        # the depot as a selectable action so the agent can go back to reload.
        if self.vehicle_position == 0:
            self.visited_mask[0] = 1
        else:
            self.visited_mask[0] = 0

        # Action mask handed to the policy: a node is blocked if it has already been
        # served, or if the remaining load cannot cover its full delivery.
        infeasible = self.replenishment_amount > self.current_load_capacity[0]
        load_mask = np.zeros(self.num_retailers + 1, dtype=int)
        load_mask[1:] = infeasible
        effective_mask = np.maximum(self.visited_mask, load_mask)


        routing_obs = {
            "location": np.vstack([self.depot_location, self.location]),
            "vehicle_position": self.vehicle_position,
            "replenishment_amount": self.replenishment_amount,
            "current_load_capacity": self.current_load_capacity,
            "visited_mask": effective_mask
        }

        r_vrp = -distance_cost * self.delivery_cost

        # Every retailer served -> close the tour and advance to the next timestep.
        if np.all(self.visited_mask[1:] == 1):
            self.current_step += 1

            # The vehicle must end each tour at the depot, so the return leg is charged
            # implicitly rather than requiring the agent to select node 0.
            r_vrp -= self._get_distance(node_1=self.location[self.vehicle_position - 1], node_2=self.depot_location[0]) * self.delivery_cost
            self.vehicle_position = 0


            # Update the historical data arrays with replenishment amounts and demands
            self.historical_demands = self._update_history_window(self.historical_demands, self.current_demand)
            self.replenishment_history = self._update_history_window(self.replenishment_history, self.replenishment_amount)
            
            # Reset routing state for the next tour: nothing visited, vehicle reloaded.
            self.visited_mask = np.zeros(self.num_retailers + 1, dtype=int)
            self.visited_mask[0] = 1
            self.current_load_capacity = np.array([self.vehicle_capacity], dtype=np.float32)
            terminated = self.current_step >= self.episode_length
            if terminated is False:
                self.current_demand = self.demand[:, self.current_step]

            truncated = False
            critic_obs = {
                "location": np.vstack([self.depot_location, self.location]),
                "current_inventory": self.retailers_current_inventory,
                "current_demand": self.current_demand,
                "holding_cost": self.holding_cost,
                "replenishment_history": self.replenishment_history,
                "historical_demands": self.historical_demands,
                "current_load_capacity": self.current_load_capacity,
                "vehicle_position": self.vehicle_position
            }

            info = {}

            return routing_obs, r_vrp, critic_obs, terminated, truncated, info
        else:
            return routing_obs, r_vrp, None, False, False, {}
        
        
    def render(self) -> None:
        """Not implemented. `route_log` is reserved for a future tour visualisation."""
        pass

    # Returns distance between two nodes
    # Euclidean distance rounded to the nearest integer, matching the rounding
    # convention used by the benchmark instances so results stay comparable.
    def _get_distance(self, node_1: npt.NDArray[np.float32], node_2: npt.NDArray[np.float32]) -> int:
        return round(np.linalg.norm(
            node_1 - node_2, ord=2
        ))

    # Updates history arrays based on movement in time
    # Sliding window of shape (num_retailers, lookback_window): shift left, drop the
    # oldest period, and write `current_value` into the newest (last) column.
    def _update_history_window(
        self, history_arr: npt.NDArray[np.float32], current_value: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        history_arr = np.roll(history_arr, shift=-1, axis=1)
        history_arr[:, -1] = current_value
        return history_arr

    # Builds the fixed graph topology consumed by the GIN layers. Currently a complete
    # graph (every node adjacent to every other, no self-loops), since the benchmark
    # instances impose no travel restrictions between nodes.
    def _create_adjacency_list(self, num_nodes: int) -> Dict[int, List[int]]:
        return {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}
        