import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *
from param import *


class GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_class):
        super().__init__()

        self.layer1 = GCNlayer(input_dim, hidden_dim)
        self.layer2 = GCNlayer(hidden_dim, num_class)

    def forward(self, x, adj_mat):
        x = F.relu(self.layer1(x, adj_mat))
        x = F.dropout(x, args.dropout, training=self.training)
        x = self.layer2(x, adj_mat)
        out = F.log_softmax(x, dim=1)

        return out