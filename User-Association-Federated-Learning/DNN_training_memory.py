import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np


class DNN_TRAINING_MEMORY():

  def __init__(self, max_size=20):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    X_inputs, y_outputs = [], []
    for i in ind:
      X_input, y_output = self.storage[i]
      X_inputs.append(np.array(X_input, copy=False))
      y_outputs.append(np.array(y_output, copy=False))
    return np.array(X_inputs), np.array(y_outputs)