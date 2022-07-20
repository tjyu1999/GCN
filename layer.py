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
        nn.init.uniform_(self.weight, a=0, b=1)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.constant_(self.bias, val=0.3)

    def forward(self, x, adj_mat):
        x = torch.matmul(x, self.weight)
        x = torch.spmm(adj_mat, x)
        out = x + self.bias

        return out
