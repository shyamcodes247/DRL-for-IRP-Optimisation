from torch.nn import Sequential, ReLU, Linear

def build_mlp(in_dim, hidden_dims, out_dim):
        """
        Builds a plain feedforward MLP: `in_dim -> hidden_dims[0] -> ... ->
        hidden_dims[-1] -> out_dim`, with ReLU activations between the
        `Linear` layers.

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
            layers += [Linear(dims[i], dims[i + 1]), ReLU()]
        layers.append(Linear(dims[-1], out_dim))
        return Sequential(*layers)