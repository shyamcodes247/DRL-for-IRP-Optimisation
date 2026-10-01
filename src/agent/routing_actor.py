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
    def __init__(self, node_feature_dim, gin_dims, mlp_dims, logit_clip=10.0):
        """
        Args:
            node_feature_dim: Dimensionality of the raw per-node input
                features, passed straight through to `GINEncoder`.
            gin_dims: List of GIN layer output dimensionalities, passed
                straight through to `GINEncoder`.
            mlp_dims: Hidden-layer sizes for `decoder`.
            logit_clip: Bound on the magnitude of a node's logit, applied as
                `logit_clip * tanh(raw)` (Kool et al., 2019, who use 10).
                Nothing else stops the logits growing: the policy gradient
                keeps sharpening a distribution that is already right, and a
                300-epoch run was measured going from a logit spread of 0.0
                to 675 — a softmax so peaked it is a hard argmax with no
                exploration left, after which tour quality stopped improving
                and then drifted back (excess 1.06x at epoch 200, 1.11x by
                300). tanh is monotone, so the ordering of preferences
                survives; only the scale is bounded.
        """
        super().__init__()
        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=gin_dims)
        self.decoder = build_mlp(2 * self.gin.output_dim, mlp_dims, 1)
        self.logit_clip = logit_clip

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

        return self.logit_clip * torch.tanh(logits)

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
            logit_spread: Python float — standard deviation of the selectable
                nodes' logits. The policy can only prefer one node over
                another by the gap between their scores, so a spread
                collapsing towards zero means the distribution is flattening
                to uniform and node choice is becoming arbitrary, whatever
                the tour costs happen to look like. Returned here rather than
                recomputed by the caller, which would cost a second forward
                pass per hop. NaN when fewer than two nodes are selectable
                (no choice to make).
        """
        logits = self.forward(node_features=node_features)
        # A fully-covering mask leaves every logit at -inf, and Categorical's
        # normalisation then computes -inf - (-inf) = NaN, which propagates
        # silently into the weights. Fail here instead, naming the cause.
        if bool((mask == 1).all()):
            raise ValueError(
                "every node is masked, so there is no legal next stop: the tour "
                "cannot close and the policy has nothing to choose from. Check "
                "IRPEnv's load feasibility mask against the period's deliveries."
            )
        logits = logits.masked_fill(mask == 1, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        selectable = logits[torch.isfinite(logits)].detach()
        logit_spread = float(selectable.std()) if selectable.numel() > 1 else float("nan")

        return action.item(), dist.log_prob(action), logit_spread

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