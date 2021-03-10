"""
Defines models for active learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import consistent_mc_dropout

from .radial_layers.variational_bayes import (
    SVI_Linear, SVIConv2D, SVIMaxPool2D)


class BNN(nn.Module):
    def __init__(self, p, channels):
        super().__init__()
        self.p = p
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5)
        self.conv2 = nn.Conv2d(channels, 2*channels, kernel_size=5)
        self.fc_in_dim = 32 * channels
        self.fc1 = nn.Linear(self.fc_in_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv1(x), p=self.p, training=True), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x), p=self.p, training=True), 2))
        x = x.view(-1, self.fc_in_dim)
        x = F.relu(F.dropout(self.fc1(x), p=self.p, training=True))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class RadialBNN(nn.Module):
    def __init__(self, channels):
        super(RadialBNN, self).__init__()
        prior = {"name": "gaussian_prior",
                 "sigma": 0.25,
                 "mu": 0}
        initial_rho = -4
        self.conv1 = SVIConv2D(1, channels, [5,5], "radial", prior, initial_rho, "he")
        self.conv2 = SVIConv2D(channels, channels * 2, [5, 5], "radial", prior, initial_rho, "he")
        self.fc_in_dim = 32 * channels
        self.fc1 = SVI_Linear(self.fc_in_dim, 128, initial_rho, "he", "radial", prior)
        self.fc2 = SVI_Linear(128, 10, initial_rho, "he", "radial", prior)
        self.maxpool = SVIMaxPool2D((2,2))

    def forward(self, x):
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        variational_samples = x.shape[1]
        x = x.view(-1, variational_samples, self.fc_in_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=2)


class TinyRadialBNN(nn.Module):
    def __init__(self):
        super(TinyRadialBNN, self).__init__()
        prior = {"name": "gaussian_prior",
                 "sigma": 0.25,
                 "mu": 0}
        initial_rho = -4
        self.fc1 = SVI_Linear(784, 50, initial_rho, "he", "radial", prior)
        self.fc2 = SVI_Linear(50, 10, initial_rho, "he", "radial", prior)

    def forward(self, x):
        variational_samples = x.shape[1]
        x = x.view(-1, variational_samples, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=2)

class LinearRadialBNN(nn.Module):
    def __init__(self):
        super(TinyRadialBNN, self).__init__()
        prior = {"name": "gaussian_prior",
                 "sigma": 0.25,
                 "mu": 0}
        initial_rho = -4
        self.fc1 = SVI_Linear(784,10, initial_rho, "he", "radial", prior)

    def forward(self, x):
        variational_samples = x.shape[1]
        x = x.view(-1, variational_samples, 784)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class ToyBNN(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(F.dropout(self.fc1(x), p=self.p, training=True))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConsistentBNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input