import torch
import torch.nn as nn


class GCNlayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.uniform_(self.weight)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj_mat):
        x = torch.matmul(x, self.weight)
        x = torch.spmm(adj_mat, x)
        out = x + self.bias

        return out