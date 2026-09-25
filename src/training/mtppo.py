import os
import sys
from typing import Any, Dict, List

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
        the m sampled instances (`train`'s env pool) once, then takes
        exactly one gradient step (`update`) over everything collected that
        epoch — not multiple minibatch passes.
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

    def _normalize_location(self, features: torch.Tensor, scale: float) -> torch.Tensor:
        """Divides the leading `loc_dim` (location) columns of `features` by `scale`."""
        features = features.clone()
        features[:, : self.loc_dim] = features[:, : self.loc_dim] / scale
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

        with torch.no_grad():
            while not terminated:
                critic_node_feats = self._normalize_location(
                    build_critic_features(critic_obs), scale
                ).to(self.device)
                critic_global_feats = build_global_features(critic_obs).to(self.device)
                value = self.critic(critic_node_feats, critic_global_feats)

                inv_obs = self._inventory_obs_from_critic(critic_obs)
                inv_node_feats = self._normalize_location(
                    build_inventory_features(inv_obs), scale
                ).to(self.device)
                inv_hist_feats = build_inventory_history(inv_obs).to(self.device)
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
                        build_routing_features(routing_obs), scale
                    ).to(self.device)
                    mask = torch.from_numpy(routing_obs["visited_mask"]).to(self.device)
                    route_action, route_logp = self.routing_actor.act(route_node_feats, mask)

                    routing_obs, r_vrp, next_critic_obs, terminated, _, _ = env.routing_action_step(
                        route_action
                    )
                    total_r_vrp += r_vrp

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

                buffer.add_timestep(
                    critic_obs=(critic_node_feats, critic_global_feats),
                    inventory_obs=inv_node_feats,
                    inventory_history=inv_hist_feats,
                    action=inv_action,
                    log_prob=inv_logp,
                    r_inv=r_inv,
                    value=value,
                    done=terminated,
                )

        return {
            "r_inv": total_r_inv,
            "r_vrp": total_r_vrp,
            "total_reward": total_r_inv + total_r_vrp,
        }

    def evaluate_episode(self, env: Any) -> Dict[str, float]:
        """
        Runs one deterministic (greedy) episode on `env`: the inventory
        actor's mean action instead of a sampled one, the routing actor's
        highest-probability (masked) node instead of a sampled one. Reports
        the paper's cost-breakdown metrics (Lu et al., 2025, Tables 4-6):
        inventory cost, delivery distance, fill rate, and their sum.

        `IRPEnv`'s demand is fixed per instance (not resampled per reset),
        so a greedy rollout is fully deterministic — one call is enough,
        there's no benefit to averaging over repeated episodes.

        Returns:
            Dict with `inv_cost` (total holding + lost-sales cost),
            `vrp_distance` (raw travel distance, undiscounted by
            `delivery_cost`), `routing_cost` (`vrp_distance * delivery_cost`,
            i.e. the paper's VRP.Dist*1k-style delivery cost term),
            `total_cost` (`inv_cost + routing_cost`), `fill_rate` (percent
            of demand served immediately from stock), and `stockout_count`.
        """
        self.critic.eval()
        self.inv_actor.eval()
        self.routing_actor.eval()

        scale = self._location_scale(env)
        _, critic_obs, _ = env.reset()

        total_inv_cost = 0.0
        total_distance = 0.0
        total_lost_units = 0.0
        total_demand = 0.0
        total_stockouts = 0
        terminated = False

        with torch.no_grad():
            while not terminated:
                inv_obs = self._inventory_obs_from_critic(critic_obs)
                inv_node_feats = self._normalize_location(build_inventory_features(inv_obs), scale).to(
                    self.device
                )
                inv_hist_feats = build_inventory_history(inv_obs).to(self.device)
                mu, _ = self.inv_actor(inv_node_feats, inv_hist_feats)

                total_demand += float(env.current_demand.sum())
                routing_obs, r_inv, info = env.inventory_action_step(mu.cpu().numpy())
                total_inv_cost += -r_inv
                total_lost_units += info["lost_sales_units"]
                total_stockouts += info["stockout_count"]

                guard = 0
                while True:
                    route_node_feats = self._normalize_location(
                        build_routing_features(routing_obs), scale
                    ).to(self.device)
                    mask = torch.from_numpy(routing_obs["visited_mask"]).to(self.device)
                    logits = self.routing_actor(route_node_feats).masked_fill(mask == 1, float("-inf"))
                    route_action = int(torch.argmax(logits).item())

                    routing_obs, r_vrp, next_critic_obs, terminated, _, _ = env.routing_action_step(
                        route_action
                    )
                    total_distance += -r_vrp / max(env.delivery_cost, 1e-12)

                    if next_critic_obs is not None:
                        critic_obs = next_critic_obs
                        break
                    guard += 1
                    assert guard < 10_000, "greedy routing loop did not terminate"

        self.critic.train()
        self.inv_actor.train()
        self.routing_actor.train()

        routing_cost = total_distance * env.delivery_cost
        fill_rate = 100.0 * (1.0 - total_lost_units / total_demand) if total_demand > 0 else 100.0

        return {
            "inv_cost": total_inv_cost,
            "vrp_distance": total_distance,
            "routing_cost": routing_cost,
            "total_cost": total_inv_cost + routing_cost,
            "fill_rate": fill_rate,
            "stockout_count": total_stockouts,
        }

    @staticmethod
    def _normalize(advantages: torch.Tensor) -> torch.Tensor:
        if advantages.numel() <= 1:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        One aggregated gradient step over everything currently in `buffer`
        (Algorithm 1, lines 16-20), then clears it. Matches the pseudocode
        literally: `L_CLIP^{i,k}` and the critic's MSE loss are each summed
        across every instance i's episode collected this epoch (lines 16-18,
        inside the i-loop), and exactly one gradient update follows (line 20,
        outside it) — not multiple minibatch passes.

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

        Returns:
            Dict of this epoch's losses/entropy, for logging.
        """
        buffer.compute_advantage(self.gamma)
        buffer.advantages = self._normalize(buffer.advantages)

        num_timesteps = len(buffer.r_inv)
        batch = next(buffer.get_batches(num_timesteps))

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

        buffer.clear()
        return {
            "policy_inv": inv_policy_loss.item(),
            "policy_vrp": vrp_policy_loss.item(),
            "value": value_loss.item(),
            "entropy": (entropy_inv.item() + entropy_vrp.item()) / 2,
        }

    def train(
        self,
        envs: List[Any],
        num_epochs: int,
        log_every: int = 1,
        on_epoch_end: Any = None,
    ) -> None:
        """
        Top-level training loop (Algorithm 1): for each epoch, roll out
        every one of the m instances in `envs` exactly once (line 3's
        "sampling m instances from a uniform distribution" — realized here
        as the fixed pool of instance files passed in, touched once per
        epoch rather than resampled with replacement) into a shared buffer,
        then run exactly one aggregated gradient update on everything
        collected (line 20 — a single step, not multiple minibatch passes).

        Args:
            envs: The m problem instances (e.g. every file in a benchmark
                subfolder). Pass a single-element list to train on one
                fixed instance.
            num_epochs: Number of outer collect+update cycles.
            log_every: Print aggregated episode stats every this many epochs.
            on_epoch_end: Optional callback `(epoch, episode_stats, losses)`
                invoked after each epoch's update, e.g. for checkpointing.
        """
        buffer = RolloutBuffer()

        for epoch in range(1, num_epochs + 1):
            episode_stats = [self.collect_episode(env, buffer) for env in envs]

            losses = self.update(buffer)

            if log_every and epoch % log_every == 0:
                mean_r_inv = sum(s["r_inv"] for s in episode_stats) / len(episode_stats)
                mean_r_vrp = sum(s["r_vrp"] for s in episode_stats) / len(episode_stats)
                print(
                    f"[epoch {epoch:5d}] "
                    f"mean r_inv={mean_r_inv:10.2f}  mean r_vrp={mean_r_vrp:10.2f}  "
                    f"policy_inv={losses['policy_inv']:.4f}  policy_vrp={losses['policy_vrp']:.4f}  "
                    f"value={losses['value']:.4f}  entropy={losses['entropy']:.4f}"
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
