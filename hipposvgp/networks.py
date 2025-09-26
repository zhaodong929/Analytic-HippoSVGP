import torch.nn as nn
import torch

class MLPEncoder(nn.Module):

    def __init__(self, dim_y, dim_f, num_hidden_units):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_y, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 2*dim_f),
        )
        self.dim_f = dim_f

    def forward(self, y):
        y = y
        out = self.net(y.float()).double()
        return out[:, :self.dim_f], torch.nn.functional.softplus(out[:, self.dim_f:])

class MLPDecoder(nn.Module):

    def __init__(self, dim_y, dim_f, num_hidden_units):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_f, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, dim_y),
        )

    def forward(self, f):
        return self.net(f.float()).double()
