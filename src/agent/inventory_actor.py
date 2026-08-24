from typing import List, Tuple
import torch
from gin import GINEncoder
from mlp import build_mlp

class InventoryActor(torch.nn.Module):
    """
        Per-retailer continuous replenishment policy. Encodes retailer node
        features with a GIN, embeds each retailer's demand/replenishment
        history separately, concatenates the two per-retailer, and decodes
        a Normal distribution's (mu, sigma) over the replenishment quantity
        for every retailer in parallel.

        NOTE: `__init__` never calls `super().__init__()`, so `nn.Module`'s
        internal state (`_parameters`, `_modules`, etc.) is never set up —
        the `self.gin = ...` assignment on the next line raises
        `AttributeError` immediately, since `nn.Module.__setattr__` relies
        on that state existing.
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
            mu: Tensor of shape (num_retailers,) — per-retailer mean
                replenishment quantity.
            sigma: Tensor of shape (num_retailers,) — per-retailer standard
                deviation, obtained via `exp` so it stays positive.
        """
        h = self.gin(node_features)
        e = self.state_embed(history_features)
        combined = torch.cat([h, e], dim=1)
        out = self.decoder(combined)
        mu, rho = out[:, 0], out[:, 1]
        sigma = torch.exp(rho)

        return mu, sigma

    def act(
        self, node_features: torch.Tensor, history_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a replenishment action from the policy for the given state.

        Args:
            node_features: As in `forward`.
            history_features: As in `forward`.

        Returns:
            action: Tensor of shape (num_retailers,) — sampled replenishment
                quantities.
            log_prob: Scalar tensor — summed log-probability of `action`
                under the per-retailer Normal distributions.
        """
        mu, sigma = self.forward(node_features=node_features, history_features=history_features)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action, log_prob

    def evaluate(
        self, node_features: torch.Tensor, history_features: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scores a given action under the current policy, for PPO-style updates.

        Args:
            node_features: As in `forward`.
            history_features: As in `forward`.
            action: Tensor of shape (num_retailers,) — the replenishment
                action to evaluate (e.g. one sampled during rollout).

        Returns:
            log_prob: Scalar tensor — summed log-probability of `action`
                under the current policy's per-retailer distributions.
            entropy: Scalar tensor — summed entropy of the per-retailer
                distributions, used as an exploration bonus.
        """
        mu, sigma = self.forward(node_features=node_features, history_features=history_features)
        dist = torch.distributions.Normal(mu, sigma)
        return dist.log_prob(action).sum(), dist.entropy().sum()