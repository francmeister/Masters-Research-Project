import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
         super(DNN, self).__init__()
         self.fc1 = nn.Linear(input_dim, 40)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(40, 30)
         self.fc3 = nn.Linear(30, output_dim)
         #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x