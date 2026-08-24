import torch
from gin import GINEncoder
from mlp import build_mlp

class Critic(torch.nn.Module):
    """
        Centralized critic for the CTDE architecture: encodes the joint
        (depot + retailer) node features with a GIN, mean-pools the node
        embeddings into a single graph-level vector, concatenates that with
        the global (non-per-node) features, and maps the result through an
        MLP head to a scalar state-value estimate.

        Global scalars tied to routing state (vehicle_position,
        load_capacity) are deliberately excluded entirely: the critic is
        evaluated on the pre-decision state, where the vehicle is always at
        the depot with a full load, so they are constant and uninformative.
        Other global scalars (e.g. depot inventory, production rate) are not
        excluded — they are fed in via `global_features` alongside the
        pooled node embedding rather than as padded per-node columns (see
        `build_global_features` in `features.py`).
    """
    def __init__(self, node_feature_dim, global_feature_dim, hidden_dims, mlps_dim):
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features (see `GINEncoder`).
            global_feature_dim: Dimensionality of the global (non-per-node)
                feature vector, concatenated onto the pooled node embedding
                before the MLP head.
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
            global_features: Tensor of shape (global_feature_dim,) — the
                non-per-node features for the same state (see
                `build_global_features`).

        Returns:
            Scalar tensor: the estimated value of the given state.
        """
        h = self.gin(node_features)
        pooled = h.mean(dim=0)
        combined = torch.cat([pooled, global_features])
        return self.head(combined)