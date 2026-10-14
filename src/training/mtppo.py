import os
import sys
import weakref
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
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
        skip_dead_zone: float = 0.1,
        routing_uses_position: bool = True,
        shared_routing_advantage: bool = False,
        upper_dead_zone: Optional[float] = None,
        inventory_choices: Optional[List[float]] = None,
        kl_coeff: float = 0.0,
        kl_target: float = 0.01,
        kl_adaptive: Optional[bool] = None,
        vf_clip_param: Optional[float] = None,
        target_kl_early_stop: Optional[float] = None,
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
            skip_dead_zone: Inventory fraction below which nothing
                discretionary is delivered, so a retailer that needs nothing
                this period can be skipped entirely (see
                `_delivery_from_fraction`). Larger values make skipping
                easier for the policy to reach, at the cost of action range.
        """
        self.loc_dim = loc_dim
        self.skip_dead_zone = skip_dead_zone
        # Ablation switches for reproducing Lu et al. (2025) as published.
        # `routing_uses_position=False` drops the displacement/is-current
        # columns from the routing features (their Eq. 30);
        # `shared_routing_advantage=True` scores routing hops with the shared
        # advantage rather than the self-critical one (their Eq. 35). Both
        # default to this project's modified behaviour.
        self.routing_uses_position = routing_uses_position
        self.shared_routing_advantage = shared_routing_advantage

        # Adaptive KL penalty and value-loss clipping (Lu et al., 2025, Table 3,
        # whose parameter names indicate RLlib's PPO). All default to inert, so
        # a run that does not ask for them reproduces earlier results exactly.
        #
        # `kl_coeff` is the INITIAL beta; the live values are per-actor and
        # mutate across `update()` calls, which is why they are instance state
        # rather than constants. They are separate because the inventory
        # actor's log-prob is summed over retailers (see
        # `InventoryActor.evaluate`), so its KL is a sum of n per-retailer KLs
        # and runs roughly n times larger than the routing actor's single
        # categorical KL. One shared beta would be driven entirely by
        # inventory and would mean nothing for routing.
        # See `_delivery_from_fraction`. None keeps the original mapping.
        self.upper_dead_zone = upper_dead_zone
        # A fixed set of fractions the inventory actor chooses between, e.g.
        # [0.0, 0.5, 1.0]. None keeps the continuous Beta. The dead zones are
        # bypassed in this mode (see `_delivery_for`): they exist to make
        # "minimum only" and "full fill" reachable from a continuous action,
        # and both are already exact atoms here.
        self.inventory_choices = inventory_choices
        self.kl_coeff_inv = float(kl_coeff)
        self.kl_coeff_vrp = float(kl_coeff)
        # Lu et al. do not state a KL target; 0.01 is RLlib's default.
        self.kl_target = float(kl_target)
        self.kl_adaptive = (kl_coeff > 0.0) if kl_adaptive is None else bool(kl_adaptive)
        self.vf_clip_param = vf_clip_param
        self.target_kl_early_stop = target_kl_early_stop
        self._routing_coord_cols = (2 if routing_uses_position else 1) * loc_dim
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
            action_values=inventory_choices,
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

        # Per-env constants, computed once rather than on every
        # `collect_episode` call. Keyed weakly rather than by `id(env)`:
        # CPython reuses an id once the object behind it is collected, so an
        # env built after another was dropped could be handed the previous
        # one's scales. With differently-sized instances that surfaces as a
        # broadcast error, and between same-sized ones it would silently
        # normalize by the wrong constants. A weak key drops with its env.
        self._loc_scale_cache: "weakref.WeakKeyDictionary[Any, float]" = weakref.WeakKeyDictionary()
        self._quantity_scale_cache: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()

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
        if env not in self._loc_scale_cache:
            coords = np.concatenate([env.location.ravel(), env.depot_location.ravel()])
            self._loc_scale_cache[env] = float(np.max(np.abs(coords))) or 1.0
        return self._loc_scale_cache[env]

    def _fractions_from_action(self, action: torch.Tensor) -> npt.NDArray:
        """
        The fractions a sampled action stands for. In discrete mode the action
        is an index, so this looks the value up; in continuous mode it already
        IS the fraction.
        """
        return self.inv_actor.action_value(action).detach().cpu().numpy()

    def _delivery_for(self, fractions: npt.NDArray, env: Any) -> npt.NDArray:
        """
        Fractions to quantities, with the dead zones applied only in
        continuous mode. A discrete value of 0 must mean exactly the minimum
        and 1 exactly a full top-up; passing them through the lower dead zone
        would leave 0 correct but compress 0.5, and the upper one is redundant
        because 1.0 is already an exact atom. `dead_zone=0.0` reduces the map
        to the plain interpolation `m + v * (headroom - m)`.
        """
        if self.inventory_choices is not None:
            return self._delivery_from_fraction(
                fractions, env.retailers_current_inventory, env.current_demand,
                env.retailer_max_capacity, 0.0, None,
            )
        return self._delivery_from_fraction(
            fractions, env.retailers_current_inventory, env.current_demand,
            env.retailer_max_capacity, self.skip_dead_zone, self.upper_dead_zone,
        )

    @staticmethod
    def _cover_classes(env: Any) -> npt.NDArray:
        """
        Each retailer's g_i = U_i / r_i, the number of periods one full top-up
        covers. Archetti et al. (2007) draw it from {2, 3}, and the two behave
        very differently: a g=2 retailer can only be skipped after an exact
        full fill, whereas a g=3 retailer has slack and can be skipped from a
        fraction around 0.55. Rounded because the ratio is computed in float.
        """
        demand = env.demand[:, 0]
        return np.round(env.retailer_max_capacity / np.maximum(demand, 1e-12)).astype(int)

    @staticmethod
    def _delivery_from_fraction(
        fraction: npt.NDArray,
        inventory: npt.NDArray,
        demand: npt.NDArray,
        max_capacity: npt.NDArray,
        dead_zone: float = 0.1,
        upper_dead_zone: Optional[float] = None,
    ) -> npt.NDArray:
        """
        Maps each retailer's policy fraction in [0, 1] onto a delivery
        quantity spanning its feasible range, so the action the
        `InventoryActor` chooses is the one the environment carries out.

            <= dead_zone -> the shortfall alone: exactly enough to avoid a
                 stockout this period, i.e. what `IRPEnv`'s auto-top-up floor
                 would have forced anyway. Where a retailer's stock already
                 covers the period the shortfall is zero, so this delivers
                 *nothing* and the retailer is not visited at all.
            1 -> a full top-up to the retailer's maximum level `U_i`

        The dead zone exists because skipping a retailer is the action that
        actually saves routing cost, and without it the policy cannot express
        it: `IRPEnv` counts a visit whenever the delivery exceeds 1e-6, and a
        Beta is continuous, so every retailer drew a positive fraction and
        every retailer was visited every period. Measured over 8 epochs with
        no dead zone: deliveries moved (55% of headroom) while visits/period
        stayed pinned at exactly the retailer count.

        Below the dead zone the discretionary amount is zero; above it, it
        ramps from zero rather than jumping, so the map stays continuous and
        the policy can approach "skip" gradually instead of having to land in
        a narrow band.

        The lower end reproduces Archetti et al. (2007)'s no-stockout
        constraint by construction, so the environment's floor never
        overrides the policy; the upper end is the order-up-to policy their
        model applies. Everything in between is the trade-off the agent gets
        to make: delivering above the shortfall costs a little holding cost
        now (0.02-0.04 per unit-period here) in exchange for skipping visits
        later, and a visit costs hundreds of distance units.

        Computed from the same `inventory`/`demand` the environment will use
        on this step, so the shortfall matches its `min_required` exactly.

        Args:
            fraction: (num_retailers,) values in [0, 1].
            inventory: (num_retailers,) current stock.
            demand: (num_retailers,) this period's demand.
            max_capacity: (num_retailers,) each retailer's `U_i`.
            dead_zone: Fraction below which nothing discretionary is
                delivered, making "skip this retailer" reachable.
            upper_dead_zone: Fraction at or above which the retailer is topped
                up to `U_i` exactly. `None` keeps the original mapping, where
                only `fraction == 1.0` fills completely.

                This exists because a full fill is the only delivery that lets
                a g=2 retailer (U_i = 2 r_i) be skipped the following period:
                it must end the period holding exactly one period of demand,
                which needs q = headroom exactly. A fraction of 0.99 leaves a
                shortfall of ~0.1 units, and `IRPEnv` counts any delivery over
                1e-6 as a visit -- so the vehicle drives out to deliver a tenth
                of a unit. Measured: the fixed-rule sweep scored f=0.999999
                identically to f=0.75 (routing cost equal to the unit) while
                f=1.0 saved 1,233 units of routing on the validation set.

                The winning action was therefore a single point of measure
                zero, which a continuous policy samples with probability zero
                -- and the Beta cannot even approach it, since beta >= 1 caps
                the mean at alpha/(alpha+1) and reaching 0.9 needs a decoder
                logit above +8. The trained seed-0 checkpoint tops out at a
                mean of 0.779 over 300 states. This widens that point into the
                interval [upper_dead_zone, 1].

        Returns:
            (num_retailers,) delivery quantities. Never below the shortfall,
            so the no-stockout constraint holds however small the fraction —
            only a retailer that needs nothing this period can be skipped.
            The environment may still scale these down together if the depot
            is short or they exceed one vehicle load.
        """
        headroom = np.maximum(max_capacity - inventory, 0.0)
        shortfall = np.clip(demand - inventory, 0.0, headroom)
        # The ramp runs from `dead_zone` up to `top`. With no upper dead zone
        # `top` is 1.0 and this is the original expression exactly; with one,
        # the ramp is compressed into [dead_zone, top] and everything at or
        # above `top` clips to a full top-up.
        top = 1.0 if upper_dead_zone is None else upper_dead_zone
        discretionary = np.clip((fraction - dead_zone) / (top - dead_zone), 0.0, 1.0)
        return shortfall + discretionary * (headroom - shortfall)

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
        if env not in self._quantity_scale_cache:
            retailer_scale = np.maximum(np.asarray(env.retailer_max_capacity, dtype=float), 1e-8)
            depot_scale = float(env.vehicle_capacity) or 1.0
            self._quantity_scale_cache[env] = (retailer_scale, depot_scale)
        return self._quantity_scale_cache[env]

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
        # Configuration, not data: survives the buffer's clear() between epochs,
        # but set here so a buffer handed to a differently-configured agent
        # cannot silently keep the previous one's scoring rule.
        buffer.shared_routing_advantage = self.shared_routing_advantage
        _, critic_obs, _ = env.reset()
        total_r_inv, total_r_vrp = 0.0, 0.0
        terminated = False
        scale = self._location_scale(env)
        retailer_scale, depot_scale = self._quantity_scales(env)
        recorder = RouteRecorder(
            np.vstack([env.depot_location, env.location]), env.delivery_cost
        )
        # How much of each retailer's headroom actually gets delivered. Pinned
        # at shortfall/headroom means the inventory actor is not in control —
        # the state the policy sat in before it chose a fraction rather than a
        # raw quantity.
        fill_fractions: List[float] = []
        requested_fractions: List[float] = []
        skip_rates: List[float] = []
        high_fraction_shares: List[float] = []
        # Visit rates split by g_i, so "did g=2 retailers start getting
        # skipped" -- the question the upper dead zone exists to answer -- is
        # directly readable instead of inferred from an aggregate.
        cover_class = self._cover_classes(env)
        visits_by_g: Dict[int, List[float]] = {2: [], 3: []}

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
                # The action is a per-retailer fraction; the buffer stores it
                # unchanged, so the PPO ratio is taken on the same variable
                # the policy actually parameterises.
                inv_action, inv_logp = self.inv_actor.act(inv_node_feats, inv_hist_feats)
                # `inv_action` is an index in discrete mode and already a
                # fraction otherwise; `_fractions_from_action` hides that, and
                # the index is what the buffer stores for PPO's ratio.
                inv_fractions = self._fractions_from_action(inv_action)
                delivery = self._delivery_for(inv_fractions, env)
                headroom = np.maximum(env.retailer_max_capacity - env.retailers_current_inventory, 0.0)

                routing_obs, r_inv, _ = env.inventory_action_step(delivery)
                total_r_inv += r_inv
                fill_fractions.append(
                    float(np.mean(env.replenishment_amount[headroom > 0] / headroom[headroom > 0]))
                    if np.any(headroom > 0) else float("nan")
                )
                requested_fractions.append(float(np.mean(inv_fractions)))
                # Share of retailers the policy chose to skip outright. Only
                # possible where the retailer needs nothing this period, so
                # this is the lever on visit count (and therefore on routing
                # cost) that the dead zone exists to make reachable.
                skip_rates.append(float(np.mean(env.replenishment_amount <= 1e-6)))
                threshold = self.upper_dead_zone if self.upper_dead_zone is not None else 1.0
                high_fraction_shares.append(float(np.mean(inv_fractions >= threshold)))
                served = env.replenishment_amount > 1e-6
                for g_value in (2, 3):
                    members = cover_class == g_value
                    if members.any():
                        visits_by_g[g_value].append(float(np.mean(served[members])))

                # This timestep's index among `add_timestep` calls so far — assigned
                # before `add_timestep` runs, since every routing hop below must be
                # tagged with it, but the timestep record itself can only be added
                # once the tour (and therefore `terminated`) is known.
                timestep_index = len(buffer.r_inv)

                while True:
                    route_node_feats = self._normalize_location(
                        build_routing_features(
                            routing_obs, retailer_scale, self.routing_uses_position
                        ),
                        scale,
                        num_coord_cols=self._routing_coord_cols,
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
            "fill_fraction": float(np.nanmean(fill_fractions)) if fill_fractions else float("nan"),
            "requested_fraction": float(np.mean(requested_fractions))
            if requested_fractions else float("nan"),
            "skip_rate": float(np.mean(skip_rates)) if skip_rates else float("nan"),
            "high_fraction_share": float(np.mean(high_fraction_shares))
            if high_fraction_shares else float("nan"),
            "visit_rate_g2": float(np.mean(visits_by_g[2])) if visits_by_g[2] else float("nan"),
            "visit_rate_g3": float(np.mean(visits_by_g[3])) if visits_by_g[3] else float("nan"),
            **recorder.summary(),
        }

    def evaluate_episode(
        self,
        env: Any,
        route_log: Optional[List[Dict[str, Any]]] = None,
        collect_diagnostics: bool = True,
    ) -> Dict[str, float]:
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
            collect_diagnostics: When False, the `RouteRecorder` and its
                2-opt reference tours are skipped entirely and every
                diagnostic field in the result is NaN. The cost figures are
                unaffected -- the policy and the environment do exactly the
                same work either way. Use it when only the solution matters
                and the diagnostics' cost would distort the measurement, as
                when timing inference: 2-opt over 50 stops dominates the
                forward passes it is meant to be measured against.
            route_log: If given, one dict per period is appended describing
                that period's tour in full — the visit sequence, the distance
                travelled, the 2-opt distance over the same stops, and the
                mean nearest-neighbour rank of its hops. Use it to inspect
                individual routing decisions; the returned metrics only carry
                episode-level means.

        Returns:
            Dict with `inv_cost` (period-0 opening-stock holding cost, plus
            holding and lost-sales cost over the horizon -- so it exceeds the
            summed `r_inv` by `env.initial_holding_cost`),
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
        if route_log is not None and not collect_diagnostics:
            raise ValueError("route_log needs collect_diagnostics=True to be populated")

        self.critic.eval()
        self.inv_actor.eval()
        self.routing_actor.eval()

        scale = self._location_scale(env)
        retailer_scale, _depot_scale = self._quantity_scales(env)
        _, critic_obs, _ = env.reset()

        # Seeded with the period-0 opening-stock holding cost rather than 0.
        # It is a per-instance constant (see `IRPEnv.initial_holding_cost`), so
        # it never appears in `r_inv` and has to be added here for the reported
        # objective to match the one Archetti et al. (2007) minimise.
        total_inv_cost = float(env.initial_holding_cost)
        total_distance = 0.0
        total_lost_units = 0.0
        total_demand = 0.0
        total_stockouts = 0
        terminated = False
        recorder = (
            RouteRecorder(np.vstack([env.depot_location, env.location]), env.delivery_cost)
            if collect_diagnostics else None
        )
        eval_fill_fractions: List[float] = []
        eval_high_fraction: List[float] = []
        cover_class = self._cover_classes(env)
        # Per period, so the report can show whether later periods are the ones
        # that stop needing visits -- which is where the saving has to come
        # from if a full fill is doing its job.
        eval_visits_by_g: Dict[int, List[float]] = {2: [], 3: []}

        with torch.no_grad():
            while not terminated:
                inv_obs = self._inventory_obs_from_critic(critic_obs)
                inv_node_feats = self._normalize_location(
                    build_inventory_features(inv_obs, retailer_scale), scale
                ).to(self.device)
                inv_hist_feats = build_inventory_history(inv_obs, retailer_scale).to(self.device)
                # The Beta's mean fraction, mapped through the same transform
                # training uses. Feeding the raw decoder output here instead
                # would evaluate a different policy than the one trained.
                mean_fraction = self.inv_actor.mean_action(inv_node_feats, inv_hist_feats)
                # In discrete mode `mean_action` returns argmax INDICES, so
                # this converts them; in continuous mode it is already the
                # Beta's mean fraction and the call is the identity.
                greedy_fractions = self._fractions_from_action(mean_fraction)
                delivery = self._delivery_for(greedy_fractions, env)
                headroom = np.maximum(env.retailer_max_capacity - env.retailers_current_inventory, 0.0)

                total_demand += float(env.current_demand.sum())
                threshold = self.upper_dead_zone if self.upper_dead_zone is not None else 1.0
                eval_high_fraction.append(float(np.mean(greedy_fractions >= threshold)))
                routing_obs, r_inv, info = env.inventory_action_step(delivery)
                served_now = env.replenishment_amount > 1e-6
                for g_value in (2, 3):
                    members = cover_class == g_value
                    if members.any():
                        eval_visits_by_g[g_value].append(float(np.mean(served_now[members])))
                if np.any(headroom > 0):
                    eval_fill_fractions.append(
                        float(np.mean(env.replenishment_amount[headroom > 0] / headroom[headroom > 0]))
                    )
                total_inv_cost += -r_inv
                total_lost_units += info["lost_sales_units"]
                total_stockouts += info["stockout_count"]

                guard = 0
                while True:
                    route_node_feats = self._normalize_location(
                        build_routing_features(
                            routing_obs, retailer_scale, self.routing_uses_position
                        ),
                        scale,
                        num_coord_cols=self._routing_coord_cols,
                    ).to(self.device)
                    mask = torch.from_numpy(routing_obs["visited_mask"]).to(self.device)
                    logits = self.routing_actor(route_node_feats).masked_fill(mask == 1, float("-inf"))
                    route_action = int(torch.argmax(logits).item())

                    if recorder is not None:
                        hop_position = int(routing_obs["vehicle_position"])
                        hop_eligible = np.flatnonzero(routing_obs["visited_mask"] == 0)
                        selectable = logits[torch.isfinite(logits)]
                        hop_spread = (
                            float(selectable.std()) if selectable.numel() > 1 else float("nan")
                        )

                    routing_obs, r_vrp, next_critic_obs, terminated, _, _ = env.routing_action_step(
                        route_action
                    )
                    total_distance += -r_vrp / max(env.delivery_cost, 1e-12)
                    if recorder is not None:
                        recorder.record_hop(
                            hop_position, hop_eligible, route_action, r_vrp, hop_spread
                        )

                    if next_critic_obs is not None:
                        critic_obs = next_critic_obs
                        break
                    guard += 1
                    assert guard < 10_000, "greedy routing loop did not terminate"
                if recorder is not None:
                    recorder.close_period(reference="two_opt")

        self.critic.train()
        self.inv_actor.train()
        self.routing_actor.train()

        routing_cost = total_distance * env.delivery_cost
        fill_rate = 100.0 * (1.0 - total_lost_units / total_demand) if total_demand > 0 else 100.0
        routing = recorder.summary() if recorder is not None else {}

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
            "fill_fraction": float(np.mean(eval_fill_fractions))
            if eval_fill_fractions else float("nan"),
            "high_fraction_share": float(np.mean(eval_high_fraction))
            if eval_high_fraction else float("nan"),
            "visit_rate_g2": float(np.mean(eval_visits_by_g[2]))
            if eval_visits_by_g[2] else float("nan"),
            "visit_rate_g3": float(np.mean(eval_visits_by_g[3]))
            if eval_visits_by_g[3] else float("nan"),
            "total_cost_best_route": total_inv_cost + best_routing_cost,
        }

    def _adapt_kl(self, beta: float, measured_kl: float) -> float:
        """
        RLlib's adaptive rule: if the policy moved more than twice the target,
        tighten the leash by 1.5x; if it barely moved, slacken it by half.

        Multiplicative, so beta decays geometrically and never reaches exactly
        zero from a positive start -- which is why `kl_coeff = 0.0` is the
        clean way to switch the penalty off, rather than a small value.
        """
        if measured_kl > 2.0 * self.kl_target:
            return beta * 1.5
        if measured_kl < 0.5 * self.kl_target:
            return beta * 0.5
        return beta

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

        # Early stopping is per actor and latches: once an actor's policy has
        # moved further than `target_kl_early_stop`, it sits out the remaining
        # passes over this batch while the others keep training.
        inv_stopped = False
        vrp_stopped = False
        kl_inv_summed = kl_inv_per_ret = kl_vrp = 0.0
        clip_frac_inv = clip_frac_vrp = vf_clip_frac = 0.0

        for _ in range(ppo_epochs):
            self.critic_optimizer.zero_grad()
            self.inv_optimizer.zero_grad()
            self.routing_optimizer.zero_grad()

            clip_inv_sum = torch.zeros((), device=self.device)
            clip_vrp_sum = torch.zeros((), device=self.device)
            value_loss_sum = torch.zeros((), device=self.device)
            entropy_inv_sum = torch.zeros((), device=self.device)
            entropy_vrp_sum = torch.zeros((), device=self.device)
            kl_inv_sum = torch.zeros((), device=self.device)
            kl_vrp_sum = torch.zeros((), device=self.device)
            kl_inv_per_ret_sum = torch.zeros((), device=self.device)
            clipped_inv = clipped_vrp = vf_clipped = 0
            num_hops = 0

            for record in batch:
                node_feats, global_feats = record["critic_obs"]
                value = self.critic(node_feats.to(self.device), global_feats.to(self.device)).squeeze(-1)

                target_return = record["return"].to(self.device)
                # One record is one sample, so `mse_loss` here IS that sample's
                # squared error. RLlib's `vf_clip_param` caps it; note `clamp`
                # has ZERO gradient above the cap, so a capped sample
                # contributes nothing rather than being softly down-weighted.
                value_err = F.mse_loss(value, target_return)
                if self.vf_clip_param is not None:
                    if value_err.item() > self.vf_clip_param:
                        vf_clipped += 1
                    value_err = torch.clamp(value_err, max=self.vf_clip_param)
                value_loss_sum = value_loss_sum + value_err

                inv = record["inventory"]
                new_logp_inv, entropy_inv = self.inv_actor.evaluate(
                    inv["node_features"].to(self.device),
                    inv["history_features"].to(self.device),
                    inv["action"].to(self.device),
                )
                adv_inv = inv["advantage"].to(self.device)
                # `old_log_prob` came from the rollout under `no_grad`, so it is
                # already a constant here -- the ratio's gradient flows only
                # through `new_logp_inv`, which is what the surrogate needs.
                logratio_inv = new_logp_inv - inv["old_log_prob"].to(self.device)
                ratio_inv = torch.exp(logratio_inv)
                clip_inv_sum = clip_inv_sum + torch.min(
                    ratio_inv * adv_inv,
                    torch.clamp(ratio_inv, 1 - self.clip_eps, 1 + self.clip_eps) * adv_inv,
                )
                entropy_inv_sum = entropy_inv_sum + entropy_inv
                # Schulman's k3 estimator of KL(old || new): (r - 1) - log r.
                # Every term is >= 0 analytically (x - 1 - log x >= 0 for
                # x > 0), unlike mean(-log r), whose terms can come out
                # negative on a small batch and spuriously halve beta. Clamped
                # at 0 only to absorb float error; a negative penalty would
                # reward divergence. Reuses the surrogate's own log-ratio, so
                # the two can never disagree about how far the policy moved.
                kl_i = torch.clamp((ratio_inv - 1.0) - logratio_inv, min=0.0)
                kl_inv_sum = kl_inv_sum + kl_i
                # The loss keeps the summed KL (consistent with the summed
                # log-prob in the ratio); beta adapts on the per-retailer mean,
                # so `kl_target` means the same thing whatever the instance size.
                kl_inv_per_ret_sum = kl_inv_per_ret_sum + kl_i / max(inv["action"].numel(), 1)
                if (ratio_inv - 1.0).abs().item() > self.clip_eps:
                    clipped_inv += 1

                for hop in record["routing"]:
                    new_logp_vrp, entropy_vrp = self.routing_actor.evaluate(
                        hop["node_features"].to(self.device),
                        hop["mask"].to(self.device),
                        torch.as_tensor(hop["action"], device=self.device),
                    )
                    adv_vrp = hop["advantage"].to(self.device)
                    logratio_vrp = new_logp_vrp - hop["old_log_prob"].to(self.device)
                    ratio_vrp = torch.exp(logratio_vrp)
                    clip_vrp_sum = clip_vrp_sum + torch.min(
                        ratio_vrp * adv_vrp,
                        torch.clamp(ratio_vrp, 1 - self.clip_eps, 1 + self.clip_eps) * adv_vrp,
                    )
                    entropy_vrp_sum = entropy_vrp_sum + entropy_vrp
                    kl_vrp_sum = kl_vrp_sum + torch.clamp(
                        (ratio_vrp - 1.0) - logratio_vrp, min=0.0
                    )
                    if (ratio_vrp - 1.0).abs().item() > self.clip_eps:
                        clipped_vrp += 1
                    num_hops += 1

            num_records = len(batch)
            num_hops = max(num_hops, 1)

            value_loss = value_loss_sum / num_records
            inv_policy_loss = -(clip_inv_sum / num_records)
            vrp_policy_loss = -(clip_vrp_sum / num_hops)
            entropy_inv = entropy_inv_sum / num_records
            entropy_vrp = entropy_vrp_sum / num_hops

            # Batch means. These overwrite each pass, so after the loop they
            # hold the FINAL pass's values -- total movement away from the
            # behaviour policy after every gradient step on this batch, which
            # is the quantity beta should react to.
            kl_inv_summed = (kl_inv_sum / num_records).item()
            kl_inv_per_ret = (kl_inv_per_ret_sum / num_records).item()
            kl_vrp = (kl_vrp_sum / num_hops).item()
            clip_frac_inv = clipped_inv / num_records
            clip_frac_vrp = clipped_vrp / num_hops
            vf_clip_frac = vf_clipped / num_records

            critic_loss = self.value_coef * value_loss
            inv_loss = inv_policy_loss - self.entropy_coef * entropy_inv
            routing_loss = vrp_policy_loss - self.entropy_coef * entropy_vrp
            # L_inv = L_inv^CLIP - beta * KL - c_e * H, minimised. The summed
            # KL is used here; see the per-retailer note above. Added only when
            # beta > 0, so an off run builds exactly the old graph.
            if self.kl_coeff_inv > 0.0:
                inv_loss = inv_loss + self.kl_coeff_inv * (kl_inv_sum / num_records)
            if self.kl_coeff_vrp > 0.0:
                routing_loss = routing_loss + self.kl_coeff_vrp * (kl_vrp_sum / num_hops)

            # A stopped actor is left out of the backward entirely rather than
            # having its step skipped, so no gradient is computed for it at all.
            total_loss = critic_loss
            if not inv_stopped:
                total_loss = total_loss + inv_loss
            if not vrp_stopped:
                total_loss = total_loss + routing_loss

            # One backward pass over the sum is equivalent to three separate
            # ones here (the three branches share no parameters), but cheaper;
            # clipping and stepping are still done per-network below.
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            if not inv_stopped:
                torch.nn.utils.clip_grad_norm_(self.inv_actor.parameters(), self.max_grad_norm)
                self.inv_optimizer.step()
            if not vrp_stopped:
                torch.nn.utils.clip_grad_norm_(self.routing_actor.parameters(), self.max_grad_norm)
                self.routing_optimizer.step()

            # Checked after stepping, so the pass that crossed the threshold
            # still counts; only the passes after it are skipped. Inventory is
            # compared on the per-retailer KL, matching how its beta adapts.
            if self.target_kl_early_stop is not None:
                if kl_inv_per_ret > self.target_kl_early_stop:
                    inv_stopped = True
                if kl_vrp > self.target_kl_early_stop:
                    vrp_stopped = True

            totals["policy_inv"] += inv_policy_loss.item()
            totals["policy_vrp"] += vrp_policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += (entropy_inv.item() + entropy_vrp.item()) / 2

        # Beta is adapted ONCE per training iteration, after every pass over
        # this batch -- RLlib's rule, and the reason it lives here rather than
        # inside the loop: beta should answer "how far did this whole update
        # move the policy", not vary between passes over the same data.
        if self.kl_adaptive:
            self.kl_coeff_inv = self._adapt_kl(self.kl_coeff_inv, kl_inv_per_ret)
            self.kl_coeff_vrp = self._adapt_kl(self.kl_coeff_vrp, kl_vrp)

        buffer.clear()
        # The per-pass losses are averaged over `ppo_epochs`; the critic
        # diagnostics describe the single rollout they were all computed
        # against, so they are reported as-is.
        return {
            **{k: v / ppo_epochs for k, v in totals.items()},
            "value_mean": value_mean,
            "value_explained_var": value_explained_var,
            "advantage_bias": advantage_bias,
            # KL/clip diagnostics are logged whether or not the penalty is on,
            # so an ordinary run still shows how far each update moves each
            # policy. These are the FINAL pass's values, not an average across
            # passes (see where they are assigned).
            "approx_kl_inv": kl_inv_summed,
            "approx_kl_inv_per_retailer": kl_inv_per_ret,
            "approx_kl_vrp": kl_vrp,
            "kl_coeff_inv": self.kl_coeff_inv,
            "kl_coeff_vrp": self.kl_coeff_vrp,
            "clip_fraction_inv": clip_frac_inv,
            "clip_fraction_vrp": clip_frac_vrp,
            "vf_clip_fraction": vf_clip_frac,
            "early_stop_inv": float(inv_stopped),
            "early_stop_vrp": float(vrp_stopped),
        }

    def train(
        self,
        envs: List[Any],
        num_epochs: int,
        ppo_epochs: int = 4,
        log_every: int = 1,
        on_epoch_end: Any = None,
        lr_decay: Optional[str] = None,
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
            lr_decay: Optional learning-rate schedule decayed over
                `num_epochs`, applied to all three optimizers alike:
                "cosine" anneals smoothly to zero, "linear" ramps down to 10%
                of the starting rate. `None` holds the rate fixed. Decay
                mostly helps late in a long run, where a rate that was right
                early keeps nudging an already-reasonable policy around.
        """
        buffer = RolloutBuffer()
        optimizers = (self.critic_optimizer, self.inv_optimizer, self.routing_optimizer)
        if lr_decay == "cosine":
            schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=num_epochs)
                          for o in optimizers]
        elif lr_decay == "linear":
            schedulers = [torch.optim.lr_scheduler.LinearLR(
                o, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
                for o in optimizers]
        elif lr_decay in (None, "none"):
            schedulers = []
        else:
            raise ValueError(f"unknown lr_decay {lr_decay!r}; expected 'cosine', 'linear' or None")

        for epoch in range(1, num_epochs + 1):
            episode_stats = [self.collect_episode(env, buffer) for env in envs]

            losses = self.update(buffer, ppo_epochs=ppo_epochs)
            for scheduler in schedulers:
                scheduler.step()

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
