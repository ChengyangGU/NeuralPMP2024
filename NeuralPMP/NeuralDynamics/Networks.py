import torch
import torch.nn as nn
import time

# NN-Simulated Systems (Type A: 1 hidden-Layer MLP)
class NNTypeA(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        # Linear Layer List
        layers = [input, hidden, output]
        self.layer1 = nn.Linear(layers[0], layers[1], bias=True, dtype=torch.float32)
        self.layer2 = nn.Linear(layers[1], layers[2], bias=True, dtype=torch.float32)
        self.layer3 = nn.Linear(layers[2], layers[3], bias=True, dtype=torch.float32)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)

        return x

# NN-Simulated Systems (Type B: 2 hidden-Layer MLP)
class NNTypeB(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        # Linear Layer List
        layers = [input, hidden, hidden, output]
        self.layer1 = nn.Linear(layers[0], layers[1], bias=True, dtype=torch.float32)
        self.layer2 = nn.Linear(layers[1], layers[2], bias=True, dtype=torch.float32)
        self.layer3 = nn.Linear(layers[2], layers[3], bias=True, dtype=torch.float32)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)

        return x

# NN-Simulated Systems (Type C: 3 hidden-Layer MLP)
class NNTypeC(nn.Module):
    def __init__(self, input, output, hidden1, hidden2):
        super().__init__()
        # Linear Layer List
        layers = [input, hidden1, hidden2, hidden1, output]
        self.layer1 = nn.Linear(layers[0], layers[1], bias=True, dtype=torch.float32)
        self.layer2 = nn.Linear(layers[1], layers[2], bias=True, dtype=torch.float32)
        self.layer3 = nn.Linear(layers[2], layers[3], bias=True, dtype=torch.float32)
        self.layer4 = nn.Linear(layers[3], layers[4], bias=True, dtype=torch.float32)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        #x = torch.sin(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.layer4(x)

        return x
