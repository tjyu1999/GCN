import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvNet(nn.Module):
    def __init__(self,
                 input,
                 in_dim,
                 hid_dim,
                 out_dim,
                 max_depth):
        super().__init__()

        self.input = input
        self.input_dim = in_dim
        self.hidden_dim = hid_dim
        self.output_dim = out_dim
        self.max_depth = max_depth

        # construct the reachable matrix which is called adjacency matrix as well
        self.reachable_edge_matrix = None

        self.prep_weights, self.prep_bias = self.init(in_dim, hid_dim, out_dim)
        self.agg_weights, self.agg_bias = self.init(in_dim, hid_dim, out_dim)

    @staticmethod
    def init(in_dim, hid_dim, out_dim):
        weights = []
        bias = []

        curr_dim = in_dim
        for hid in hid_dim:
            weights.append(torch.FloatTensor((curr_dim, hid)))
            bias.append(torch.FloatTensor(hid))
            curr_dim = hid
        weights.append(torch.FloatTensor(curr_dim, out_dim))
        bias.append(torch.FloatTensor(out_dim))

        for l in range(len(hid_dim)):
            nn.init.xavier_uniform_(weights[l], gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(bias[l], val=0.5)

        return weights, bias

    def forward(self):
        x = self.input
        for l in range(len(self.prep_weights)):
            x = torch.matmul(x, self.prep_weights[l])
            x += self.prep_bias[l]
            x = F.leaky_relu(x)
        for d in range(self.max_depth):
            y = x
            for l in range(len(self.agg_weights)):
                y = torch.matmul(y, self.agg_weights[l])
                y += self.agg_bias[l]
                y = F.leaky_relu(y)
            y = torch.matmul(self.reachable_edge_matrix[d], y)
            x = x + y

        return x
