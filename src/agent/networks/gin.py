import torch
import tensorflow as tf
from keras import Sequential
from torch.nn import Linear, ReLU
from bnlearn import BN
from torch_geometric.nn import GINConv, global_mean_pool

class GINEncoder(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dims, num_layers):
        pass
    
    def forward(self, h):
        pass

