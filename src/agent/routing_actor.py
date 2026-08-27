import torch 
from torch.nn import Sequential, Linear, ReLU
from gin import GINEncoder
from mlp import build_mlp

class RoutingActor(torch.nn.Module):
    def __init__(self, broadcast, node_feature_dim, gin_dims, mlp_dims):
        super().__init__()
        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=gin_dims)
        in_dim = 2 * self.gin.output_dim if broadcast else self.gin.output_dim
        self.decoder = build_mlp(in_dim, mlp_dims, 1)
        
    def forward(self, node_features):
        h = self.gin(node_features)
        pooled = h.mean(dim=0, keepdim=True)
        combined = torch.cat([h, pooled.expand(h.shape[0], -1)], dim=1)
        logits = self.decoder(combined).squeeze(-1)
        
        return logits
    
    def act(self, node_features, mask):
        logits = self.forward(node_features=node_features)
        logits = logits.masked_fill(mask == 1, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, node_features, mask, action):
        logits = self.forward(node_features=node_features)
        logits = logits.masked_fill(mask == 1, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()