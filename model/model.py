import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


class BaseEmbeddingMLP(nn.Module):
    def __init__(self, station_size, n_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=station_size, embedding_dim=n_dim)
        self.fc1 = nn.Linear(16+n_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, e):
        e = self.embedding(e)
        x = torch.cat((x, e), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)
