import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np


class DNN_TRAINING_MEMORY():

  def __init__(self, max_size=40):
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
    X_inputs, y_outputs, sample_rewards = [], [], []
    for i in ind:
      X_input, y_output, sample_reward = self.storage[i]
      X_inputs.append(np.array(X_input, copy=False))
      y_outputs.append(np.array(y_output, copy=False))
      sample_rewards.append(np.array(sample_reward, copy=False))
    return np.array(X_inputs), np.array(y_outputs), np.array(sample_rewards)