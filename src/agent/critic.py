import torch
from gin import GINEncoder
from torch.nn import Sequential, ReLU, Linear

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
    def __init__(self, node_feature_dim, hidden_dims, mlps_dim):
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features (see `GINEncoder`).
            hidden_dims: List of GIN layer output dimensionalities, passed
                straight through to `GINEncoder`.
            mlps_dim: Intended hidden-layer sizes for the value-estimation
                head. NOTE: currently unused — `build_mlp` is called with
                `hidden_dims` (the GIN's dims) instead, so the head's depth
                is coupled to the GIN's rather than configured by this
                parameter.
        """
        super().__init__()

        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=hidden_dims)
        self.head = self.build_mlp(self.gin.output_dim, hidden_dims, 1)

    def forward(self, node_features):
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
        return self.head(pooled)


    def build_mlp(self, in_dim, hidden_dims, out_dim):
        """
        Builds a plain feedforward MLP: `in_dim -> hidden_dims[0] -> ... ->
        hidden_dims[-1] -> out_dim`, with ReLU activations between the
        `Linear` layers.

        NOTE: `layers += [Linear(...), ReLU]` appends the `ReLU` *class*
        itself rather than an instance (`ReLU()`) — `nn.Sequential` requires
        module instances, so this raises at construction time as written.

        Args:
            in_dim: Input feature dimensionality (the GIN's `output_dim`).
            hidden_dims: Hidden-layer sizes.
            out_dim: Output dimensionality (1, for a scalar value estimate).

        Returns:
            An `nn.Sequential` stack of alternating `Linear`/`ReLU` layers.
        """
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers += [Linear(dims[i], dims[i + 1]), ReLU]
        layers.append(Linear(dims[-1], out_dim))
        return Sequential(*layers)