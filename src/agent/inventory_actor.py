from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from gin import GINEncoder
from mlp import build_mlp

# Keeps a sampled fraction off the exact endpoints, where a Beta with
# alpha, beta > 1 has zero density and log_prob would diverge.
_EDGE = 1e-6


class InventoryActor(torch.nn.Module):
    """
        Per-retailer continuous replenishment policy. Encodes retailer node
        features with a GIN, embeds each retailer's demand/replenishment
        history separately, concatenates the two per-retailer, and decodes a
        Beta distribution over a *fraction* for every retailer in parallel.

        The fraction indexes each retailer's feasible delivery range rather
        than naming a quantity directly (see
        `MTPPO._delivery_from_fraction`): 0 delivers exactly the shortfall
        needed to avoid a stockout this period, 1 tops the retailer up to its
        maximum level `U_i`. Every action is therefore feasible by
        construction and already satisfies Archetti et al. (2007)'s
        no-stockout constraint, so `IRPEnv`'s auto-top-up floor never has to
        override the policy.

        This replaces an unbounded Normal over raw quantities, under which
        the decision was provably vestigial: the environment clipped any
        request up to the shortfall for free, while any excess only added
        holding cost, so "request nothing" was optimal and the trained policy
        duly converged there — inventory cost was identical to the cent
        across checkpoints and whole runs. Choosing where to sit between
        just-in-time and a full top-up is the trade-off Archetti's order-up-to
        policy exploits to buy fewer visits, and it is what the routing cost
        mostly depends on.

        Beta rather than a squashed Gaussian so the support is exactly [0, 1]
        with no change-of-variables term: the policy is over the fraction,
        and the map onto a quantity is a deterministic function of the state,
        so PPO's ratios stay exact.
    """
    def __init__(
        self,
        node_feature_dim: int,
        history_dim: int,
        gin_dims: List[int],
        mlp_dims: List[int],
        embed_dim: int,
        action_values: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features, passed straight through to `GINEncoder`.
            history_dim: Dimensionality of a retailer's flattened history
                feature vector (see `build_inventory_history` in
                `features.py`), consumed by `state_embed`.
            gin_dims: List of GIN layer output dimensionalities, passed
                straight through to `GINEncoder`.
            mlp_dims: Hidden-layer sizes shared by both `state_embed` and
                `decoder`.
            embed_dim: Output width of `state_embed`, i.e. how much of the
                decoder's input comes from history vs. from the GIN.
            action_values: When given, the policy becomes a Categorical over
                this fixed set of fractions instead of a Beta over [0, 1].
                `None` keeps the Beta.

                The discrete form exists because the single most useful
                action -- a full top-up, fraction exactly 1.0 -- has measure
                zero under any continuous density, so a Beta samples it with
                probability zero and its mean cannot reach it either (beta >= 1
                caps the mean at alpha/(alpha+1); a trained checkpoint topped
                out at 0.779 over 300 states). That matters because a
                retailer with U_i = 2 r_i can only be skipped the following
                period after an exact full fill: at fraction 0.999999 it is
                left short by ~1e-5 units, and the environment counts any
                delivery above 1e-6 as a visit. Measured on the fixed-rule
                sweep, f=0.999999 cost the same as f=0.75 while f=1.0 saved
                1,233 units of routing. As an atom with its own probability,
                1.0 becomes reachable by sampling and selectable by argmax.
        """
        super().__init__()
        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=gin_dims)
        self.state_embed = build_mlp(history_dim, mlp_dims, embed_dim)
        self.action_values: Optional[torch.Tensor]
        if action_values is None:
            self.action_values = None
            self.decoder = build_mlp(self.gin.output_dim + embed_dim, mlp_dims, 2)
        else:
            # Registered as a buffer so it moves with `.to(device)` and is
            # saved in the state dict -- a checkpoint that forgot which values
            # its indices meant would be silently misinterpreted on reload.
            self.register_buffer("action_values", torch.as_tensor(list(action_values),
                                                                  dtype=torch.float32))
            self.decoder = build_mlp(self.gin.output_dim + embed_dim, mlp_dims,
                                     len(action_values))
            # Zero bias on the output layer so every value starts equally
            # likely: the agent then samples a full top-up from the first epoch
            # instead of having to drift into it. PyTorch's default would give
            # each option a small arbitrary standing preference.
            torch.nn.init.zeros_(self.decoder[-1].bias)

    @property
    def is_discrete(self) -> bool:
        return self.action_values is not None

    def _distribution(
        self, node_features: torch.Tensor, history_features: torch.Tensor
    ) -> torch.distributions.Distribution:
        """
        The per-retailer action distribution: a Categorical over
        `action_values` when discrete, otherwise the Beta. Shared by `act`,
        `mean_action` and `evaluate` so all three can never disagree about
        what the policy is.
        """
        out = self.forward(node_features=node_features, history_features=history_features)
        if self.is_discrete:
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Beta(*out)

    def forward(
        self, node_features: torch.Tensor, history_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: Tensor of shape (num_retailers, node_feature_dim)
                — retailer node features for the GIN.
            history_features: Tensor of shape (num_retailers, history_dim)
                — per-retailer history features for `state_embed`.

        Returns:
            alpha: Tensor of shape (num_retailers,) — first Beta shape
                parameter per retailer.
            beta: Tensor of shape (num_retailers,) — second Beta shape
                parameter per retailer.
        """
        h = self.gin(node_features)
        e = self.state_embed(history_features)
        combined = torch.cat([h, e], dim=1)
        out = self.decoder(combined)
        if self.is_discrete:
            # (num_retailers, num_action_values) logits; no transform, since a
            # Categorical is already normalised by its softmax.
            return out
        # softplus keeps both parameters positive; the +1 floor keeps the
        # Beta unimodal, which rules out the U-shaped regime whose density
        # piles up at 0 and 1 and whose samples would be almost purely
        # bang-bang (Chou et al., 2017).
        alpha = F.softplus(out[:, 0]) + 1.0
        beta = F.softplus(out[:, 1]) + 1.0

        return alpha, beta

    def act(
        self, node_features: torch.Tensor, history_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a replenishment action from the policy for the given state.

        Args:
            node_features: As in `forward`.
            history_features: As in `forward`.

        Returns:
            action: Tensor of shape (num_retailers,) — sampled fractions in
                [0, 1], one per retailer. Map to quantities with
                `MTPPO._delivery_from_fraction`; this is the variable the
                rollout buffer stores and `evaluate` re-scores, so the PPO
                ratio is taken on it.
            log_prob: Scalar tensor — summed log-probability of `action`
                under the per-retailer Beta distributions.
        """
        dist = self._distribution(node_features, history_features)
        if self.is_discrete:
            # The sampled INDEX is the action, because that is the variable the
            # policy assigns probability to -- PPO's ratio has to be taken on
            # it. `action_value` turns indices into fractions for the delivery
            # map, which is a deterministic function of the state and so does
            # not enter the ratio.
            action = dist.sample()
            return action, dist.log_prob(action).sum()
        action = dist.sample().clamp(_EDGE, 1.0 - _EDGE)
        return action, dist.log_prob(action).sum()

    def mean_action(self, node_features: torch.Tensor, history_features: torch.Tensor) -> torch.Tensor:
        """
        The distribution's mean fraction per retailer, for deterministic
        (greedy) rollouts. Evaluation must use this rather than the raw
        decoder output, so that training and evaluation apply the same
        fraction-to-quantity map.
        """
        if self.is_discrete:
            # argmax over values, not a mean: averaging {0, 0.5, 1} would
            # produce a fraction the policy cannot actually choose, and the
            # whole point of the discrete form is that 1.0 is selectable.
            return self.forward(node_features=node_features,
                                history_features=history_features).argmax(dim=-1)
        alpha, beta = self.forward(node_features=node_features, history_features=history_features)
        return alpha / (alpha + beta)

    def evaluate(
        self, node_features: torch.Tensor, history_features: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores a given action under the current policy, for PPO-style updates.

        Args:
            node_features: As in `forward`.
            history_features: As in `forward`.
            action: Tensor of shape (num_retailers,) — the fractions to
                evaluate (as sampled by `act` during rollout).

        Returns:
            log_prob: Scalar tensor — summed log-probability of `action`
                under the current policy's per-retailer distributions.
            entropy: Scalar tensor — summed entropy of the per-retailer
                distributions, used as an exploration bonus.
        """
        dist = self._distribution(node_features, history_features)
        if self.is_discrete:
            return dist.log_prob(action).sum(), dist.entropy().sum()
        action = action.clamp(_EDGE, 1.0 - _EDGE)
        return dist.log_prob(action).sum(), dist.entropy().sum()

    def action_value(self, action: torch.Tensor) -> torch.Tensor:
        """
        Maps sampled action indices onto their fractions. The identity in
        continuous mode, so callers need not branch.
        """
        if not self.is_discrete:
            return action
        return self.action_values[action.long()]