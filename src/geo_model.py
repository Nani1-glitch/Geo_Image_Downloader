import torch.nn as nn
import torch

class LatLonToEmbedding(nn.Module):
    def __init__(self, input_dim=2, hidden_dim1=128, hidden_dim2=256, output_dim=128):
        super(LatLonToEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
