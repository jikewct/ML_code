import torch
import torch.nn as nn

class OneHiddenLayerModel(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 100, output_dim = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x):
        y = self.net(x)
        return y


class SimpleMultiLayerModel(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 100, output_dim = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x):
        y = self.net(x)
        return y
    