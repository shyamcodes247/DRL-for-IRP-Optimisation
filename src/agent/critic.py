import torch
from gin import GINEncoder
from torch.nn import Sequential, ReLU, Linear

class Critic(torch.nn.Module):
    # Global scalars (vehicle_position, load_capacity) are deliberately excluded:
    # the critic is evaluated on the pre-decision state, where the vehicle is
    # always at the depot with full load, so they are constant and uninformative.
    def __init__(self, node_feature_dim, hidden_dims, mlps_dim):
        super().__init__()

        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=hidden_dims)
        self.head = self.build_mlp(self.gin.output_dim, hidden_dims, 1)

    def forward(self, node_features):
        h = self.gin(node_features)
        pooled = h.mean(dim=0)
        return self.head(pooled)


    def build_mlp(self, in_dim, hidden_dims, out_dim):
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers += [Linear(dims[i], dims[i + 1]), ReLU]
        layers.append(Linear(dims[-1], out_dim))
        return Sequential(*layers)