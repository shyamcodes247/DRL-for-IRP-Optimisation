import torch 
from torch.nn import Sequential, Linear, ReLU
from gin import GINEncoder
from mlp import build_mlp

class InventoryActor(torch.nn.Module):
    def __init__(self, node_feature_dim, history_dim, gin_dims, mlp_dims, embed_dim):
        self.gin = GINEncoder(node_feature_dim=node_feature_dim, hidden_dims=gin_dims)
        self.state_embed = build_mlp(history_dim, mlp_dims, embed_dim)
        self.decoder = build_mlp(self.gin.output_dim + embed_dim, mlp_dims, 2)
        
    def forward(self, node_features, history_features):
        h = self.gin(node_features)
        e = self.state_embed(history_features)
        combined = torch.cat([h, e], dim=1)
        out = self.decoder(combined)
        mu, rho = out[:, 0], out[:, 1]
        sigma = torch.exp(rho)
        
        return mu, sigma
    
    def act(self, node_features, history_features):
        mu, sigma = self.forward(node_features=node_features, history_features=history_features)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        return action, log_prob
    
    def evaluate(self, node_features, history_features, action):
        mu, sigma = self.forward(node_features=node_features, history_features=history_features)
        dist = torch.distributions.Normal(mu, sigma)
        return dist.log_prob(action).sum(), dist.entropy().sum()