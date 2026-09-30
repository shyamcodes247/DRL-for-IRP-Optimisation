import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..")
_AGENT_DIR = os.path.join(_SRC_DIR, "agent")
for _p in (_SRC_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from training.rollout_buffer import RolloutBuffer
from training.routing_diagnostics import RouteRecorder


class MTPPO:
    """
        Multi-Task Proximal Policy Optimization (Lu et al., 2025): a two-actor,
        one-critic CTDE algorithm for the IRP-VMI. The inventory actor and
        routing actor are trained independently (Eqs. 36-37, per-task clipped
        surrogate objectives), while a single shared critic (Eq. 34) values
        the joint pre-decision state and baselines both tasks' advantages
        (see `RolloutBuffer`'s NOTE on Eq. 35).

        This class owns the three networks, each with its own optimizer (see
        `__init__`'s note on why they're kept separate rather than pooled).
        Following Algorithm 1 literally: each epoch rolls out every one of
        the m sampled instances (`train`'s env pool) once under a fixed
        policy, computes one shared advantage for everything collected, then
        takes `ppo_epochs` ("g" in the pseudocode) gradient-update passes
        over that same fixed batch before moving to the next epoch (lines
        20-32) — standard PPO's multi-pass-per-rollout update, not a single
        gradient step per epoch.
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        history_dim: int,
        global_feature_dim: int,
        gin_dims: List[int],
        mlp_dims: List[int],
        embed_dim: int,
        loc_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.9,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            node_feature_dims: Per-node input feature widths for each
                network, keyed "critic", "inventory", "routing" (see
                `features.py`'s `build_*_features` for how each is derived).
            history_dim: Width of a retailer's flattened history vector fed
                to `InventoryActor.state_embed` (see `build_inventory_history`).
            global_feature_dim: Width of the critic's global feature vector
                (see `build_global_features`).
            gin_dims: GIN layer output dimensionalities, shared by all three
                networks' encoders.
            mlp_dims: Shared hidden-layer sizes for the decoder/head MLPs.
            embed_dim: Output width of `InventoryActor.state_embed`.
            loc_dim: Dimensionality of a node's location feature. Location is
                always the leading `loc_dim` columns of every feature tensor
                built by `features.py`'s `build_*_features`, which is what
                `_normalize_location` relies on to rescale it in place.
            lr: Adam learning rate for the combined actor+critic parameters.
            gamma: Discount factor used by `RolloutBuffer.compute_advantage`.
            clip_eps: PPO clipping parameter (epsilon in Eq. 36).
            value_coef: Weight on the critic's MSE loss in the combined loss.
            entropy_coef: Weight on the entropy bonus (exploration) in the
                combined loss.
            max_grad_norm: Global gradient-norm clip applied before each
                optimizer step.
            device: torch device string the networks and batches are moved to.
        """
        self.loc_dim = loc_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.critic = Critic(
            node_feature_dim=node_feature_dims["critic"],
            global_feature_dim=global_feature_dim,
            hidden_dims=list(gin_dims),
            mlps_dim=list(mlp_dims),
        ).to(self.device)

        self.inv_actor = InventoryActor(
            node_feature_dim=node_feature_dims["inventory"],
            history_dim=history_dim,
            gin_dims=list(gin_dims),
            mlp_dims=list(mlp_dims),
            embed_dim=embed_dim,
        ).to(self.device)

        self.routing_actor = RoutingActor(
            node_feature_dim=node_feature_dims["routing"],
            gin_dims=list(gin_dims),
            mlp_dims=list(mlp_dims),
        ).to(self.device)

        # Separate optimizers (and, in `update`, separate grad-norm clipping) per
        # network: the paper describes the two actors as "independently trained"
        # with a shared critic evaluating both, and in practice this also stops
        # the critic's much larger loss scale (its regression target is a raw
        # cumulative-cost return, easily orders of magnitude bigger than a
        # clipped-and-advantage-normalized policy loss) from dominating a pooled
        # gradient norm and rescaling the actors' otherwise-healthy gradients
        # down to near zero — or, if the critic ever produces a non-finite
        # gradient, from poisoning the actors' gradients via that shared norm.
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.inv_optimizer = torch.optim.Adam(self.inv_actor.parameters(), lr=lr)
        self.routing_optimizer = torch.optim.Adam(self.routing_actor.parameters(), lr=lr)

        # Caches each env's location-scale constant (see `_location_scale`) so it
        # is computed once per env rather than on every `collect_episode` call.
        self._loc_scale_cache: Dict[int, float] = {}
        # Same, for the quantity scales (see `_quantity_scales`).
        self._quantity_scale_cache: Dict[int, Any] = {}

    @staticmethod
    def _inventory_obs_from_critic(critic_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derives an inventory-actor-shaped observation from a critic
        observation. Used for every timestep after the first, since
        `IRPEnv.routing_action_step` only returns a joint `critic_obs` on
        tour close, not a standalone inventory observation — but the two
        share the same underlying fields (the critic's `location` just has
        an extra leading depot row).
        """
        return {
            "location": critic_obs["location"][1:],
            "current_inventory": critic_obs["current_inventory"],
            "current_demand": critic_obs["current_demand"],
            "replenishment_history": critic_obs["replenishment_history"],
            "historical_demands": critic_obs["historical_demands"],
        }

    def _location_scale(self, env: Any) -> float:
        """
        Per-env coordinate scale, used by `_normalize_location` to rescale
        location features into a small, network-friendly range.

        Benchmark instance files are not guaranteed to use the paper's
        assumed (0,1) coordinate range (e.g. this repo's
        `Instances_lowcost_H6` set uses raw coordinates up to several
        hundred). Left unscaled, these feed straight into the GIN's
        neighbour-sum aggregation (`GINEncoder.forward`), which compounds
        the scale further across layers — verified to drive `sigma` in
        `InventoryActor` to ~1e-27 even at random initialization, and to
        NaN within the first PPO update once gradients start flowing.
        Cached per env (keyed by `id`) since it depends only on the
        instance's fixed node coordinates.
        """
        key = id(env)
        if key not in self._loc_scale_cache:
            coords = np.concatenate([env.location.ravel(), env.depot_location.ravel()])
            self._loc_scale_cache[key] = float(np.max(np.abs(coords))) or 1.0
        return self._loc_scale_cache[key]

    def _quantity_scales(self, env: Any):
        """
        Per-env divisors putting quantity features on a common footing (see
        `features._scaled`): each retailer's maximum inventory level `U_i`
        for per-retailer quantities, and the vehicle capacity for depot-level
        ones. Both are fixed instance parameters, so they are cached per env.

        Returns:
            (retailer_scale, depot_scale) — an array of shape
            (num_retailers,) and a scalar.
        """
        key = id(env)
        if key not in self._quantity_scale_cache:
            retailer_scale = np.maximum(np.asarray(env.retailer_max_capacity, dtype=float), 1e-8)
            depot_scale = float(env.vehicle_capacity) or 1.0
            self._quantity_scale_cache[key] = (retailer_scale, depot_scale)
        return self._quantity_scale_cache[key]

    def _normalize_location(
        self, features: torch.Tensor, scale: float, num_coord_cols: Optional[int] = None
    ) -> torch.Tensor:
        """
        Divides the leading coordinate columns of `features` by `scale`.

        Args:
            features: Per-node feature tensor from one of `features.py`'s
                `build_*_features`.
            scale: Per-env coordinate scale (see `_location_scale`).
            num_coord_cols: How many leading columns hold coordinates.
                Defaults to `loc_dim` (one location block). Routing features
                pass `2 * loc_dim`, since they carry a location *and* a
                displacement-from-current-node block, both on the raw
                coordinate scale (see `build_routing_features`).
        """
        cols = self.loc_dim if num_coord_cols is None else num_coord_cols
        features = features.clone()
        features[:, :cols] = features[:, :cols] / scale
        return features

    def collect_episode(self, env: Any, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Runs one full episode on `env` under the current policy (no
        gradient tracking) and appends every timestep/routing hop to
        `buffer`. Mirrors Algorithm 1, lines 4-15, for a single instance.

        Args:
            env: An `IRPEnv`-like environment (see `IRPEnv`'s interaction
                loop docstring).
            buffer: Rollout storage to append this episode's transitions to.
                Not cleared here — the caller decides when to clear/update.

        Returns:
            Dict with this episode's summed inventory reward, routing
            reward, and their total (all as plain floats, for logging).
        """
        _, critic_obs, _ = env.reset()
        total_r_inv, total_r_vrp = 0.0, 0.0
        terminated = False
        scale = self._location_scale(env)
        retailer_scale, depot_scale = self._quantity_scales(env)
        recorder = RouteRecorder(
            np.vstack([env.depot_location, env.location]), env.delivery_cost
        )

        with torch.no_grad():
            while not terminated:
                critic_node_feats = self._normalize_location(
                    build_critic_features(critic_obs, retailer_scale, depot_scale), scale
                ).to(self.device)
                critic_global_feats = build_global_features(critic_obs, depot_scale).to(self.device)
                value = self.critic(critic_node_feats, critic_global_feats)

                inv_obs = self._inventory_obs_from_critic(critic_obs)
                inv_node_feats = self._normalize_location(
                    build_inventory_features(inv_obs, retailer_scale), scale
                ).to(self.device)
                inv_hist_feats = build_inventory_history(inv_obs, retailer_scale).to(self.device)
                inv_action, inv_logp = self.inv_actor.act(inv_node_feats, inv_hist_feats)

                routing_obs, r_inv, _ = env.inventory_action_step(inv_action.cpu().numpy())
                total_r_inv += r_inv

                # This timestep's index among `add_timestep` calls so far — assigned
                # before `add_timestep` runs, since every routing hop below must be
                # tagged with it, but the timestep record itself can only be added
                # once the tour (and therefore `terminated`) is known.
                timestep_index = len(buffer.r_inv)

                while True:
                    route_node_feats = self._normalize_location(
                        build_routing_features(routing_obs, retailer_scale),
                        scale,
                        num_coord_cols=2 * self.loc_dim,
                    ).to(self.device)
                    mask = torch.from_numpy(routing_obs["visited_mask"]).to(self.device)
                    route_action, route_logp, logit_spread = self.routing_actor.act(
                        route_node_feats, mask
                    )

                    hop_position = int(routing_obs["vehicle_position"])
                    hop_eligible = np.flatnonzero(routing_obs["visited_mask"] == 0)

                    routing_obs, r_vrp, next_critic_obs, terminated, _, _ = env.routing_action_step(
                        route_action
                    )
                    total_r_vrp += r_vrp
                    recorder.record_hop(
                        hop_position, hop_eligible, route_action, r_vrp, logit_spread
                    )

                    buffer.add_routing_step(
                        routing_obs=route_node_feats,
                        routing_mask=mask,
                        action=route_action,
                        log_prob=route_logp,
                        r_vrp=r_vrp,
                        timestep_index=timestep_index,
                    )

                    if next_critic_obs is not None:
                        critic_obs = next_critic_obs
                        break
                # Closed here, before the timestep record is added, because
                # the tour's self-critical score is what every routing hop in
                # it gets trained against.
                recorder.close_period()

                buffer.add_timestep(
                    critic_obs=(critic_node_feats, critic_global_feats),
                    inventory_obs=inv_node_feats,
                    inventory_history=inv_hist_feats,
                    action=inv_action,
                    log_prob=inv_logp,
                    r_inv=r_inv,
                    value=value,
                    done=terminated,
                    routing_advantage=recorder.last_advantage(),
                )

        return {
            "r_inv": total_r_inv,
            "r_vrp": total_r_vrp,
            "total_reward": total_r_inv + total_r_vrp,
            **recorder.summary(),
        }

    def evaluate_episode(self, env: Any, route_log: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Runs one deterministic (greedy) episode on `env`: the inventory
        actor's mean action instead of a sampled one, the routing actor's
        highest-probability (masked) node instead of a sampled one. Reports
        the paper's cost-breakdown metrics (Lu et al., 2025, Tables 4-6):
        inventory cost, delivery distance, fill rate, and their sum.

        `IRPEnv`'s demand is fixed per instance (not resampled per reset),
        so a greedy rollout is fully deterministic — one call is enough,
        there's no benefit to averaging over repeated episodes.

        Args:
            env: The instance to evaluate on.
            route_log: If given, one dict per period is appended describing
                that period's tour in full — the visit sequence, the distance
                travelled, the 2-opt distance over the same stops, and the
                mean nearest-neighbour rank of its hops. Use it to inspect
                individual routing decisions; the returned metrics only carry
                episode-level means.

        Returns:
            Dict with `inv_cost` (total holding + lost-sales cost),
            `vrp_distance` (raw travel distance, undiscounted by
            `delivery_cost`), `routing_cost` (`vrp_distance * delivery_cost`,
            i.e. the paper's VRP.Dist*1k-style delivery cost term),
            `total_cost` (`inv_cost + routing_cost`), `fill_rate` (percent
            of demand served immediately from stock), and `stockout_count`,
            plus routing diagnostics (see `RouteRecorder`):
            `visits_per_period`, `vrp_distance_2opt` (the same stops in
            2-opt order), `vrp_excess_ratio` (how many times longer the
            travelled tours were), `nn_rank` (0 = always took the nearest
            eligible stop, 0.5 = indistinguishable from random), and
            `total_cost_best_route` (this episode's cost with its routing
            replaced by the 2-opt tours, inventory decisions unchanged).
        """
        self.critic.eval()
        self.inv_actor.eval()
        self.routing_actor.eval()

        scale = self._location_scale(env)
        retailer_scale, _depot_scale = self._quantity_scales(env)
        _, critic_obs, _ = env.reset()

        total_inv_cost = 0.0
        total_distance = 0.0
        total_lost_units = 0.0
        total_demand = 0.0
        total_stockouts = 0
        terminated = False
        recorder = RouteRecorder(
            np.vstack([env.depot_location, env.location]), env.delivery_cost
        )

        with torch.no_grad():
            while not terminated:
                inv_obs = self._inventory_obs_from_critic(critic_obs)
                inv_node_feats = self._normalize_location(
                    build_inventory_features(inv_obs, retailer_scale), scale
                ).to(self.device)
                inv_hist_feats = build_inventory_history(inv_obs, retailer_scale).to(self.device)
                mu, _ = self.inv_actor(inv_node_feats, inv_hist_feats)

                total_demand += float(env.current_demand.sum())
                routing_obs, r_inv, info = env.inventory_action_step(mu.cpu().numpy())
                total_inv_cost += -r_inv
                total_lost_units += info["lost_sales_units"]
                total_stockouts += info["stockout_count"]

                guard = 0
                while True:
                    route_node_feats = self._normalize_location(
                        build_routing_features(routing_obs, retailer_scale),
                        scale,
                        num_coord_cols=2 * self.loc_dim,
                    ).to(self.device)
                    mask = torch.from_numpy(routing_obs["visited_mask"]).to(self.device)
                    logits = self.routing_actor(route_node_feats).masked_fill(mask == 1, float("-inf"))
                    route_action = int(torch.argmax(logits).item())

                    hop_position = int(routing_obs["vehicle_position"])
                    hop_eligible = np.flatnonzero(routing_obs["visited_mask"] == 0)
                    selectable = logits[torch.isfinite(logits)]
                    hop_spread = float(selectable.std()) if selectable.numel() > 1 else float("nan")

                    routing_obs, r_vrp, next_critic_obs, terminated, _, _ = env.routing_action_step(
                        route_action
                    )
                    total_distance += -r_vrp / max(env.delivery_cost, 1e-12)
                    recorder.record_hop(
                        hop_position, hop_eligible, route_action, r_vrp, hop_spread
                    )

                    if next_critic_obs is not None:
                        critic_obs = next_critic_obs
                        break
                    guard += 1
                    assert guard < 10_000, "greedy routing loop did not terminate"
                recorder.close_period(reference="two_opt")

        self.critic.train()
        self.inv_actor.train()
        self.routing_actor.train()

        routing_cost = total_distance * env.delivery_cost
        fill_rate = 100.0 * (1.0 - total_lost_units / total_demand) if total_demand > 0 else 100.0
        routing = recorder.summary()

        # What this same episode would have cost if its stops had been visited
        # in 2-opt order instead. The inventory decisions (and therefore which
        # retailers get served, and all holding/stockout cost) are held fixed,
        # so the gap between `total_cost` and `total_cost_best_route` is purely
        # the price of the routing actor's sequencing.
        reference_distance = routing.get("total_reference", float("nan"))
        best_routing_cost = reference_distance * env.delivery_cost

        if route_log is not None:
            for period, record in enumerate(recorder.periods):
                route_log.append({
                    "period": period,
                    "visits": int(record["visits"]),
                    "distance": record["travelled"],
                    "distance_2opt": record["reference"],
                    "excess_ratio": record["ratio"],
                    "nn_rank": record["nn_rank"],
                    "sequence": " ".join(str(n) for n in record["sequence"]),
                })

        return {
            "inv_cost": total_inv_cost,
            "vrp_distance": total_distance,
            "routing_cost": routing_cost,
            "total_cost": total_inv_cost + routing_cost,
            "fill_rate": fill_rate,
            "stockout_count": total_stockouts,
            "visits_per_period": routing.get("visits_per_period", float("nan")),
            "vrp_distance_2opt": reference_distance,
            "vrp_excess_ratio": routing.get("tour_ratio", float("nan")),
            "nn_rank": routing.get("nn_rank", float("nan")),
            "logit_spread": routing.get("logit_spread", float("nan")),
            "total_cost_best_route": total_inv_cost + best_routing_cost,
        }

    def update(self, buffer: RolloutBuffer, ppo_epochs: int) -> Dict[str, float]:
        """
        Runs `ppo_epochs` gradient-update passes ("g" in Algorithm 1, lines
        20-32) over everything currently in `buffer`, then clears it. The
        shared advantage (line 17) is computed once, up front, from the
        value estimates captured at collection time under the fixed policy
        that produced this rollout — standard PPO: each of the `ppo_epochs`
        passes below re-evaluates every record's log-probs/entropy/value
        under the *current* (evolving within this call) network weights and
        takes one gradient step, all measured against that same fixed
        advantage and the stored `old_log_prob` from collection (line 23's
        `θ_k^old` — no separately copied network is needed, since the
        stored old log-prob already *is* the only quantity a ratio needs
        from the old policy).

        The critic is re-evaluated on every timestep's joint state, the
        inventory actor on its single per-timestep action, and the routing
        actor on every hop taken during that timestep's tour — every term
        (critic's regression target, inventory's clipped surrogate, every
        routing hop's clipped surrogate) uses the *same* shared per-timestep
        return/advantage (see `RolloutBuffer`'s class NOTE): Eq. (35)/(36)
        write `Â^t` with no per-task subscript, so both actors are trained
        against one combined (holding + stockout + routing) signal rather
        than each seeing only its own reward stream — the inventory actor's
        gradient reflects the routing-cost consequences of its replenishment
        decisions too. Multiple routing sub-actions within one timestep
        share that timestep's single value, per Fig. 3's decomposition.

        Args:
            buffer: Rollout storage populated by one `collect_episode` call
                per instance this epoch (see `train`).
            ppo_epochs: Number of gradient-update passes over this epoch's
                fixed rollout (line 20's `g` range).

        Returns:
            Dict of this epoch's losses/entropy, averaged over the
            `ppo_epochs` passes, plus critic diagnostics for the rollout they
            were computed against. NOTE `value` is the critic's mean squared
            error, not its output, and it lives on the *normalized*-return
            scale (see `RolloutBuffer.compute_advantage`), not the raw cost
            scale — so it is small by construction and is not comparable to
            `r_inv`/`r_vrp`. `value_mean` is what the critic actually
            predicts, `value_explained_var` how much of the return's variance
            it captures, and `advantage_bias` the residual's mean before
            re-centring.
        """
        buffer.compute_advantage(self.gamma)

        # Re-centre the advantage across the epoch's batch. `compute_advantage`
        # normalizes the two reward *streams* (Algorithm 1, line 12), but the
        # resulting advantage still carries whatever systematic offset the
        # critic's bias leaves behind: measured at -0.282 (against a spread of
        # 1.229) on a trained checkpoint, with 59.8% of timesteps negative. An
        # offset like that is a near-uniform "everything you did was worse than
        # expected" signal, which pushes every action's probability down rather
        # than discriminating between them. Scaling to unit variance as well
        # keeps the step size comparable across epochs as the reward
        # distribution shifts.
        # Critic diagnostics, captured before the advantage is re-centred (at
        # this point `advantages` is exactly the regression residual,
        # return - V). The logged `value` is an MSE, which says nothing about
        # what the critic predicts or whether the prediction is any good, so
        # report both directly:
        #   value_mean          - what V(s) actually outputs, on the
        #                         normalized-return scale it is trained on
        #                         (not the raw cost scale)
        #   value_explained_var - share of the return's variance V captures;
        #                         1.0 perfect, 0 no better than predicting the
        #                         mean, < 0 actively worse than that
        #   advantage_bias      - mean of return - V before re-centring. A
        #                         persistent non-zero value is a near-uniform
        #                         "everything you did was worse than expected"
        #                         signal; it is what the re-centring below
        #                         removes, and worth watching for drift.
        advantages = buffer.advantages
        returns = buffer.returns
        if advantages.numel() > 1:
            return_var = torch.var(returns)
            value_mean = (returns - advantages).mean().item()
            advantage_bias = advantages.mean().item()
            value_explained_var = (
                (1.0 - torch.var(advantages) / return_var).item()
                if return_var > 0 else float("nan")
            )
            buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            value_mean = advantage_bias = value_explained_var = float("nan")

        num_timesteps = len(buffer.r_inv)
        batch = next(buffer.get_batches(num_timesteps))

        totals = {"policy_inv": 0.0, "policy_vrp": 0.0, "value": 0.0, "entropy": 0.0}

        for _ in range(ppo_epochs):
            self.critic_optimizer.zero_grad()
            self.inv_optimizer.zero_grad()
            self.routing_optimizer.zero_grad()

            clip_inv_sum = torch.zeros((), device=self.device)
            clip_vrp_sum = torch.zeros((), device=self.device)
            value_loss_sum = torch.zeros((), device=self.device)
            entropy_inv_sum = torch.zeros((), device=self.device)
            entropy_vrp_sum = torch.zeros((), device=self.device)
            num_hops = 0

            for record in batch:
                node_feats, global_feats = record["critic_obs"]
                value = self.critic(node_feats.to(self.device), global_feats.to(self.device)).squeeze(-1)

                target_return = record["return"].to(self.device)
                value_loss_sum = value_loss_sum + F.mse_loss(value, target_return)

                inv = record["inventory"]
                new_logp_inv, entropy_inv = self.inv_actor.evaluate(
                    inv["node_features"].to(self.device),
                    inv["history_features"].to(self.device),
                    inv["action"].to(self.device),
                )
                adv_inv = inv["advantage"].to(self.device)
                ratio_inv = torch.exp(new_logp_inv - inv["old_log_prob"].to(self.device))
                clip_inv_sum = clip_inv_sum + torch.min(
                    ratio_inv * adv_inv,
                    torch.clamp(ratio_inv, 1 - self.clip_eps, 1 + self.clip_eps) * adv_inv,
                )
                entropy_inv_sum = entropy_inv_sum + entropy_inv

                for hop in record["routing"]:
                    new_logp_vrp, entropy_vrp = self.routing_actor.evaluate(
                        hop["node_features"].to(self.device),
                        hop["mask"].to(self.device),
                        torch.as_tensor(hop["action"], device=self.device),
                    )
                    adv_vrp = hop["advantage"].to(self.device)
                    ratio_vrp = torch.exp(new_logp_vrp - hop["old_log_prob"].to(self.device))
                    clip_vrp_sum = clip_vrp_sum + torch.min(
                        ratio_vrp * adv_vrp,
                        torch.clamp(ratio_vrp, 1 - self.clip_eps, 1 + self.clip_eps) * adv_vrp,
                    )
                    entropy_vrp_sum = entropy_vrp_sum + entropy_vrp
                    num_hops += 1

            num_records = len(batch)
            num_hops = max(num_hops, 1)

            value_loss = value_loss_sum / num_records
            inv_policy_loss = -(clip_inv_sum / num_records)
            vrp_policy_loss = -(clip_vrp_sum / num_hops)
            entropy_inv = entropy_inv_sum / num_records
            entropy_vrp = entropy_vrp_sum / num_hops

            critic_loss = self.value_coef * value_loss
            inv_loss = inv_policy_loss - self.entropy_coef * entropy_inv
            routing_loss = vrp_policy_loss - self.entropy_coef * entropy_vrp

            # One backward pass over the sum is equivalent to three separate
            # ones here (the three branches share no parameters), but cheaper;
            # clipping and stepping are still done per-network below.
            (critic_loss + inv_loss + routing_loss).backward()

            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.inv_actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.routing_actor.parameters(), self.max_grad_norm)

            self.critic_optimizer.step()
            self.inv_optimizer.step()
            self.routing_optimizer.step()

            totals["policy_inv"] += inv_policy_loss.item()
            totals["policy_vrp"] += vrp_policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += (entropy_inv.item() + entropy_vrp.item()) / 2

        buffer.clear()
        # The per-pass losses are averaged over `ppo_epochs`; the critic
        # diagnostics describe the single rollout they were all computed
        # against, so they are reported as-is.
        return {
            **{k: v / ppo_epochs for k, v in totals.items()},
            "value_mean": value_mean,
            "value_explained_var": value_explained_var,
            "advantage_bias": advantage_bias,
        }

    def train(
        self,
        envs: List[Any],
        num_epochs: int,
        ppo_epochs: int = 4,
        log_every: int = 1,
        on_epoch_end: Any = None,
    ) -> None:
        """
        Top-level training loop (Algorithm 1): for each epoch, roll out
        every one of the m instances in `envs` exactly once under the
        current (fixed-for-this-epoch) policy (line 3's "sampling m
        instances from a uniform distribution" — realized here as the fixed
        pool of instance files passed in, touched once per epoch rather than
        resampled with replacement) into a shared buffer, then run
        `ppo_epochs` gradient-update passes over everything collected (lines
        20-32's "g" loop) before moving to the next epoch.

        Args:
            envs: The m problem instances (e.g. every file in a benchmark
                subfolder). Pass a single-element list to train on one
                fixed instance.
            num_epochs: Number of outer collect+update cycles.
            ppo_epochs: Number of gradient-update passes per epoch's
                collected rollout (see `update`).
            log_every: Print aggregated episode stats every this many epochs.
            on_epoch_end: Optional callback `(epoch, episode_stats, losses)`
                invoked after each epoch's update, e.g. for checkpointing.
        """
        buffer = RolloutBuffer()

        for epoch in range(1, num_epochs + 1):
            episode_stats = [self.collect_episode(env, buffer) for env in envs]

            losses = self.update(buffer, ppo_epochs=ppo_epochs)

            if log_every and epoch % log_every == 0:
                mean_r_inv = sum(s["r_inv"] for s in episode_stats) / len(episode_stats)
                mean_r_vrp = sum(s["r_vrp"] for s in episode_stats) / len(episode_stats)
                print(
                    f"[epoch {epoch:5d}] "
                    f"mean r_inv={mean_r_inv:10.2f}  mean r_vrp={mean_r_vrp:10.2f}  "
                    f"policy_inv={losses['policy_inv']:.4f}  policy_vrp={losses['policy_vrp']:.4f}  "
                    f"value_mse={losses['value']:.4f}  V_ev={losses['value_explained_var']:.3f}  "
                    f"adv_bias={losses['advantage_bias']:+.3f}  entropy={losses['entropy']:.4f}"
                )

            if on_epoch_end is not None:
                on_epoch_end(epoch, episode_stats, losses)

    def save(self, path: str) -> None:
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "inv_actor": self.inv_actor.state_dict(),
                "routing_actor": self.routing_actor.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "inv_optimizer": self.inv_optimizer.state_dict(),
                "routing_optimizer": self.routing_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.inv_actor.load_state_dict(checkpoint["inv_actor"])
        self.routing_actor.load_state_dict(checkpoint["routing_actor"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.inv_optimizer.load_state_dict(checkpoint["inv_optimizer"])
        self.routing_optimizer.load_state_dict(checkpoint["routing_optimizer"])
