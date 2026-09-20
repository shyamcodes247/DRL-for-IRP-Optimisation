from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import torch

class RolloutBuffer:
    """
        On-policy rollout storage for MTPPO's two-actor, one-critic setup
        (Lu et al., 2025). One entry is added per environment timestep for the
        inventory actor and the shared critic (`add_timestep`), and one entry
        per routing hop within that timestep's tour (`add_routing_step`),
        since a tour can take a variable number of hops to serve every
        retailer (see `IRPEnv.routing_action_step`). Each routing entry is
        tagged with the index of the timestep it belongs to, so its reward
        can be folded into that timestep's totals and its advantage can be
        looked up after `compute_advantage` runs.

        NOTE on advantage estimation: Eq. (35) of the paper is not directly
        implementable as written (`+ gamma^(T-1) * V(s^t)` reduces to
        rescaling the same baseline being subtracted, and doesn't depend on
        `t`, which cannot be the intended bootstrap term). This is
        implemented as the standard n-step return baselined against the
        critic's own value estimate instead: for each task k (inv/vrp),
        `Â^t_k = (r^t_k + gamma*r^{t+1}_k + ... ) - V(s^t)`, restarting the
        discounted sum at episode boundaries (`done`). `V(s^t)` is the single
        shared critic value (Eq. 34 — one critic for the joint state), so
        both tasks' advantages are baselined against the same value but use
        each task's own reward stream, matching how Algorithm 1 (lines 8-9)
        keeps `r^t_inv` and `r^t_vrp` as separate per-timestep sums.

        GINEncoder (`gin.py`) has no batched-graph support — it consumes one
        graph's (num_nodes, feature_dim) tensor at a time — so `get_batches`
        yields plain lists of per-timestep records rather than stacked
        tensors; the training loop is expected to loop over each record.
    """
    def __init__(self) -> None:
        self.critic_obs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.inventory_obs: List[torch.Tensor] = []
        self.inventory_history: List[torch.Tensor] = []
        self.inventory_actions: List[torch.Tensor] = []
        self.inventory_logprobs: List[torch.Tensor] = []
        self.r_inv: List[float] = []
        self.values: List[torch.Tensor] = []
        self.done: List[bool] = []

        self.routing_obs: List[torch.Tensor] = []
        self.routing_masks: List[torch.Tensor] = []
        self.routing_actions: List[int] = []
        self.routing_logprobs: List[torch.Tensor] = []
        self.r_vrp: List[float] = []
        self.timestep_index: List[int] = []

        self.returns_inv: torch.Tensor = torch.empty(0)
        self.returns_vrp: torch.Tensor = torch.empty(0)
        self.adv_inv: torch.Tensor = torch.empty(0)
        self.adv_vrp: torch.Tensor = torch.empty(0)

    def add_timestep(
        self,
        critic_obs: Tuple[torch.Tensor, torch.Tensor],
        inventory_obs: torch.Tensor,
        inventory_history: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        r_inv: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Records one environment timestep's inventory decision and the
        critic's value estimate for the pre-decision joint state.

        Args:
            critic_obs: (node_features, global_features) pair fed to
                `Critic.forward` for this timestep (see `build_critic_features`
                / `build_global_features` in `features.py`).
            inventory_obs: As returned by `build_inventory_features`, the
                per-retailer node features fed to `InventoryActor`.
            inventory_history: As returned by `build_inventory_history`.
            action: The sampled replenishment action, from `InventoryActor.act`.
            log_prob: The action's log-probability under the policy that
                sampled it (the "old" policy for this transition's PPO ratio).
                Detached before storing, since it must stay fixed across the
                PPO epochs that follow.
            r_inv: Inventory reward for this timestep (`r_inv` returned by
                `IRPEnv.inventory_action_step`).
            value: The critic's `V(critic_obs)` for this timestep. Detached
                before storing for the same reason as `log_prob`.
            done: Whether this timestep's episode terminated before starting
                the next one (see `IRPEnv.routing_action_step`'s `terminated`).
        """
        self.critic_obs.append(critic_obs)
        self.inventory_obs.append(inventory_obs)
        self.inventory_history.append(inventory_history)
        self.inventory_actions.append(action)
        self.inventory_logprobs.append(log_prob.detach())
        self.r_inv.append(r_inv)
        self.values.append(value.detach())
        self.done.append(done)

    def add_routing_step(
        self,
        routing_obs: torch.Tensor,
        routing_mask: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        r_vrp: float,
        timestep_index: int,
    ) -> None:
        """
        Records one routing hop (a single node visited) within a timestep's
        tour. A tour spans a variable number of hops, so this is called once
        per hop rather than once per timestep.

        Args:
            routing_obs: As returned by `build_routing_features` for the
                state the hop was chosen from.
            routing_mask: The `visited_mask` (effective mask) the hop was
                chosen under (see `RoutingActor.act`).
            action: The node index chosen, from `RoutingActor.act`.
            log_prob: The action's log-probability under the policy that
                sampled it. Detached before storing, as in `add_timestep`.
            r_vrp: Routing reward for this hop (`r_vrp` returned by
                `IRPEnv.routing_action_step`).
            timestep_index: Index into this timestep's position among the
                `add_timestep` calls so far — links the hop back to the
                timestep whose tour it belongs to, so `compute_advantage` can
                fold hop rewards into that timestep's total and the hop can
                later be looked up with that timestep's advantage.
        """
        self.routing_obs.append(routing_obs)
        self.routing_masks.append(routing_mask)
        self.routing_actions.append(action)
        self.routing_logprobs.append(log_prob.detach())
        self.r_vrp.append(r_vrp)
        self.timestep_index.append(timestep_index)

    def compute_advantage(self, gamma: float) -> None:
        """
        Computes per-timestep discounted returns and advantages for both
        actors (see the class NOTE for the estimator used). Must be called
        after a rollout is fully collected and before `get_batches`.

        Hop-level `r_vrp` rewards are first summed by `timestep_index` into
        one total per timestep (mirroring `r^t_vrp` in Algorithm 1, line 9),
        so the routing return/advantage series lines up 1:1 with
        `r_inv`/`values`/`done` — the same alignment every routing hop is
        looked up against in `get_batches`.

        Args:
            gamma: Discount factor.
        """
        num_timesteps = len(self.r_inv)
        r_vrp_per_timestep = np.zeros(num_timesteps)
        for reward, t in zip(self.r_vrp, self.timestep_index):
            r_vrp_per_timestep[t] += reward

        self.returns_inv, self.adv_inv = self._discounted_returns_and_advantages(self.r_inv, gamma)
        self.returns_vrp, self.adv_vrp = self._discounted_returns_and_advantages(
            r_vrp_per_timestep.tolist(), gamma
        )

    def get_batches(self, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Yields shuffled minibatches of per-timestep records for PPO epochs.
        `compute_advantage` must be called first.

        Each record bundles everything needed to replay one timestep through
        both actors and the critic: the critic's observation/return for that
        timestep, the inventory actor's single decision, and every routing
        hop taken during that timestep's tour (looked up via
        `timestep_index`; a "routing" list of length 0 means the tour closed
        without needing a hop this timestep, e.g. two decisions landing back
        to back). As noted on the class, a minibatch is a plain list — the
        training loop should iterate it and accumulate loss per record
        rather than expect stacked tensors.

        Args:
            batch_size: Number of timesteps per minibatch (the last
                minibatch may be smaller).

        Yields:
            Lists of records shaped as:
                {
                    "critic_obs": (node_features, global_features),
                    "return_inv": scalar tensor,
                    "return_vrp": scalar tensor,
                    "inventory": {
                        "node_features", "history_features", "action",
                        "old_log_prob", "advantage",
                    },
                    "routing": [
                        {"node_features", "mask", "action", "old_log_prob",
                         "advantage"},
                        ...
                    ],
                }
        """
        num_timesteps = len(self.r_inv)
        routing_by_timestep: List[List[Dict[str, Any]]] = [[] for _ in range(num_timesteps)]
        for i, t in enumerate(self.timestep_index):
            routing_by_timestep[t].append({
                "node_features": self.routing_obs[i],
                "mask": self.routing_masks[i],
                "action": self.routing_actions[i],
                "old_log_prob": self.routing_logprobs[i],
                "advantage": self.adv_vrp[t],
            })

        order = np.random.permutation(num_timesteps)
        for start in range(0, num_timesteps, batch_size):
            batch_timesteps = order[start:start + batch_size]
            batch = []
            for t in batch_timesteps:
                batch.append({
                    "critic_obs": self.critic_obs[t],
                    "return_inv": self.returns_inv[t],
                    "return_vrp": self.returns_vrp[t],
                    "inventory": {
                        "node_features": self.inventory_obs[t],
                        "history_features": self.inventory_history[t],
                        "action": self.inventory_actions[t],
                        "old_log_prob": self.inventory_logprobs[t],
                        "advantage": self.adv_inv[t],
                    },
                    "routing": routing_by_timestep[t],
                })
            yield batch

    def clear(self) -> None:
        """Empties the buffer, e.g. after using its data for a PPO update."""
        self.critic_obs = []
        self.inventory_obs = []
        self.inventory_history = []
        self.inventory_actions = []
        self.inventory_logprobs = []
        self.r_inv = []
        self.values = []
        self.done = []

        self.routing_obs = []
        self.routing_masks = []
        self.routing_actions = []
        self.routing_logprobs = []
        self.r_vrp = []
        self.timestep_index = []

        self.returns_inv = torch.empty(0)
        self.returns_vrp = torch.empty(0)
        self.adv_inv = torch.empty(0)
        self.adv_vrp = torch.empty(0)

    def _discounted_returns_and_advantages(
        self, rewards: List[float], gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rewards: One reward per timestep, aligned with `self.values` /
                `self.done`.
            gamma: Discount factor.

        Returns:
            returns: Tensor of shape (num_timesteps,) — the discounted
                return-to-go from each timestep, resetting at `done`
                boundaries. Used as the critic's regression target.
            advantages: `returns` minus the critic's value estimate at each
                timestep (`self.values`).
        """
        returns = [0.0] * len(rewards)
        running = 0.0
        for i in reversed(range(len(rewards))):
            if self.done[i]:
                running = 0.0
            running = rewards[i] + gamma * running
            returns[i] = running

        returns = torch.as_tensor(returns, dtype=torch.float32)
        values = torch.as_tensor(self.values, dtype=torch.float32)
        advantages = returns - values

        return returns, advantages
