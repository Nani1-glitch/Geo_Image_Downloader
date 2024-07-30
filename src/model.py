import torch
import torch.nn as nn
import torch.nn.functional as F

class GeographyAwareModel(nn.Module):
    def __init__(self):
        super(GeographyAwareModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 64 * 64, 128)  # Adjust dimensions based on your input size
        self.fc2 = nn.Linear(128, 2)  # Output layer for latitude and longitude

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = GeographyAwareModel()
    print(model)
