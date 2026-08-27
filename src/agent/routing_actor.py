import torch
from torch.nn import Sequential, Linear, ReLU
from gin import GINEncoder
from mlp import build_mlp

class RoutingActor(torch.nn.Module):
    """
        Pointer-network-style routing policy. Encodes the joint (depot +
        retailer) node set with a GIN, broadcasts a mean-pooled graph-level
        embedding onto every node so each node's score can be made in the
        context of the whole graph, and decodes a scalar logit per node — the
        unnormalised preference for visiting that node next.
    """
    def __init__(self, node_feature_dim, gin_dims, mlp_dims):
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features, passed straight through to `GINEncoder`.
            gin_dims: List of GIN layer output dimensionalities, passed
                straight through to `GINEncoder`.
            mlp_dims: Hidden-layer sizes for `decoder`.
        """
        super().__init__()
        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=gin_dims)
        self.decoder = build_mlp(2 * self.gin.output_dim, mlp_dims, 1)

    def forward(self, node_features):
        """
        Args:
            node_features: Tensor of shape (num_nodes, node_feature_dim) —
                depot + retailer node features (see `build_routing_features`
                in `features.py`).

        Returns:
            Tensor of shape (num_nodes,): per-node logits scoring each node
            as the next node to visit.
        """
        h = self.gin(node_features)
        pooled = h.mean(dim=0, keepdim=True)
        combined = torch.cat([h, pooled.expand(h.shape[0], -1)], dim=1)
        logits = self.decoder(combined).squeeze(-1)

        return logits

    def act(self, node_features, mask):
        """
        Samples the next node to visit from the policy for the given state.

        Args:
            node_features: As in `forward`.
            mask: Tensor of shape (num_nodes,) — 1 for nodes that are
                invalid to visit next (e.g. already visited), 0 for valid
                ones. Masked nodes get a logit of -inf so they are never
                sampled.

        Returns:
            action: Python int — index of the sampled next node.
            log_prob: Scalar tensor — log-probability of `action` under the
                masked distribution.
        """
        logits = self.forward(node_features=node_features)
        logits = logits.masked_fill(mask == 1, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def evaluate(self, node_features, mask, action):
        """
        Scores a given action under the current policy, for PPO-style updates.

        Args:
            node_features: As in `forward`.
            mask: As in `act`.
            action: Tensor holding the node index to evaluate (e.g. one
                sampled during rollout).

        Returns:
            log_prob: Scalar tensor — log-probability of `action` under the
                current policy's masked distribution.
            entropy: Scalar tensor — entropy of the masked distribution,
                used as an exploration bonus.
        """
        logits = self.forward(node_features=node_features)
        logits = logits.masked_fill(mask == 1, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()