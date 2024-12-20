import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(32 * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 10) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) 
        x = self.pool(torch.relu(self.conv2(x))) 
        x = x.view(-1, 32 * 8 * 8) 
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x) 
        x = self.softmax(x)
        return x
