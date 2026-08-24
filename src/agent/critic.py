import torch
from gin import GINEncoder
from mlp import build_mlp

class Critic(torch.nn.Module):
    """
        Centralized critic for the CTDE architecture: encodes the joint
        (depot + retailer) node features with a GIN, mean-pools the node
        embeddings into a single graph-level vector, and maps that through
        an MLP head to a scalar state-value estimate.

        Global scalars (vehicle_position, load_capacity) are deliberately
        excluded from the node features consumed here: the critic is
        evaluated on the pre-decision state, where the vehicle is always at
        the depot with a full load, so they are constant and uninformative.
    """
    def __init__(self, node_feature_dim, global_feature_dim, hidden_dims, mlps_dim):
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features (see `GINEncoder`).
            hidden_dims: List of GIN layer output dimensionalities, passed
                straight through to `GINEncoder`.
            mlps_dim: Intended hidden-layer sizes for the value-estimation
                head.
        """
        super().__init__()

        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=hidden_dims)
        self.head = build_mlp(self.gin.output_dim + global_feature_dim, mlps_dim, 1)

    def forward(self, node_features, global_features):
        """
        Args:
            node_features: Tensor of shape (num_nodes, node_feature_dim) —
                the critic's joint (depot + retailer) node features for one
                pre-decision state.

        Returns:
            Scalar tensor: the estimated value of the given state.
        """
        h = self.gin(node_features)
        pooled = h.mean(dim=0)
        combined = torch.cat([pooled, global_features])
        return self.head(combined)