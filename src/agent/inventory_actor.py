from typing import List, Tuple
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
        """
        super().__init__()
        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=gin_dims)
        self.state_embed = build_mlp(history_dim, mlp_dims, embed_dim)
        self.decoder = build_mlp(self.gin.output_dim + embed_dim, mlp_dims, 2)

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
        alpha, beta = self.forward(node_features=node_features, history_features=history_features)
        dist = torch.distributions.Beta(alpha, beta)
        action = dist.sample().clamp(_EDGE, 1.0 - _EDGE)
        log_prob = dist.log_prob(action).sum()

        return action, log_prob

    def mean_action(self, node_features: torch.Tensor, history_features: torch.Tensor) -> torch.Tensor:
        """
        The distribution's mean fraction per retailer, for deterministic
        (greedy) rollouts. Evaluation must use this rather than the raw
        decoder output, so that training and evaluation apply the same
        fraction-to-quantity map.
        """
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
        alpha, beta = self.forward(node_features=node_features, history_features=history_features)
        dist = torch.distributions.Beta(alpha, beta)
        action = action.clamp(_EDGE, 1.0 - _EDGE)
        return dist.log_prob(action).sum(), dist.entropy().sum()