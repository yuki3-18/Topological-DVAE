from __future__ import print_function
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from topologylayer.functional.utils_dionysus import *

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(729, 200)
        self.fc2 = nn.Linear(200, 24)
        self.fc3 = nn.Linear(24, 200)
        self.fc4 = nn.Linear(200, 729)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, 729))
        return self.decode(z)

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(729, 200)
        self.fc21 = nn.Linear(200, latent_dim)
        self.fc22 = nn.Linear(200, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 729)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 729))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar