from typing import List

import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

class GINEncoder(torch.nn.Module):
    """
        Graph Isomorphism Network (GIN) encoder (Xu et al., 2019) used to embed
        the IRP node set (depot + retailers) before it is consumed by the
        inventory/routing actors and critic.

        The graph is treated as complete (fully connected): every node
        aggregates over every other node with no adjacency matrix, since in
        the IRP all retailers are mutually reachable from the depot and from
        each other. Each layer updates a node's representation as

            h' = MLP((1 + eps) * h + sum_{u != v} h_u)

        where the sum runs over all other nodes' features from the previous
        layer. `eps` is a learnable per-layer scalar (initialised to 0)
        controlling how much a node weighs its own features against the
        aggregated neighbourhood.
    """
    def __init__(self, node_feature_dim: int, hidden_dims: List[int]) -> None:
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features (e.g. location, inventory level, demand history).
            hidden_dims: List of output dimensionalities, one per GIN layer.
                len(hidden_dims) is the number of layers; hidden_dims[-1]
                becomes `output_dim`, the embedding size fed downstream.
        """
        super().__init__()

        num_layers = len(hidden_dims)
        dims = [node_feature_dim] + hidden_dims
        self.mlps = torch.nn.ModuleList()

        for i in range(num_layers):
            self.mlps.append(
                Sequential(
                    Linear(dims[i], dims[i + 1]),
                    ReLU(),
                    Linear(dims[i + 1], dims[i + 1]),
                )
            )

        self.eps = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]
        )

        self.output_dim = hidden_dims[-1]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature tensor of shape (num_nodes, node_feature_dim).
                Row 0 is the depot; rows 1..num_retailers are the retailers,
                matching the node indexing convention used elsewhere in the
                environment (see `IRPEnv`).

        Returns:
            Tensor of shape (num_nodes, output_dim): the per-node embeddings
            after `len(hidden_dims)` rounds of GIN aggregation.
        """
        for i in range(len(self.mlps)):
            neighbour_sum = h.sum(dim=0, keepdim=True) - h
            combined = (1 + self.eps[i])*h + neighbour_sum
            h = self.mlps[i](combined)

        return h
