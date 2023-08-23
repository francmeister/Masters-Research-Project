import numpy as np
import gym
import Network_Env
import pybullet_envs
import torch
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
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
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x

RPB = ReplayBuffer()
state = []
state.append(1)
state.append(1)
state.append(1)
state.append(1)
next_state = []
next_state.append(2)
next_state.append(2)
next_state.append(2)
next_state.append(2)
action = []
action.append(3)
action.append(3)
action.append(3)
action.append(3)
action.append(3)
action.append(3)
action.append(3)
t = action[2]
print(t)
reward = 4
done = 5
RPB.add((state,next_state,action,reward,done))
RPB.add((state,next_state,action,reward,done))
RPB.add((state,next_state,action,reward,done))
RPB.add((state,next_state,action,reward,done))
RPB.add((state,next_state,action,reward,done))

state = np.array(state)
action = np.array(action)
Actor1 = Actor(state.shape[0],action.shape[0],4) 

print("State dim")
print(state.shape)
print("Action dim")
print(action.shape)
print("RPB")
print(RPB.storage)

batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = RPB.sample(5)
print("batch_states")
print(batch_states)
print("batch_next_states")
print(batch_next_states)
print("batch_actions")
print(batch_actions)
print("batch_rewards")
print(batch_rewards)
print("batch_dones")
print(batch_dones)

state1 = torch.Tensor(batch_states.reshape(1, -1))
print("state1")
print(state1)
batch_states = torch.Tensor(batch_states)
print("Actor output on batch_states")
print(Actor1.forward(batch_states))


h = [3,3,3,3,3,3,3,3,3,3,3,3]

print("lenght of h: ", len(h))

index = 0
f = []
for i in range(0,len(h),4):
  print("i",i)
  q,w,e,r = h[index],h[index+1],h[index+2],h[index+3]
  f.append((q,w,e,r))
  index+=4

print("f",f)


print('***************************************************************')

h = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print(h)
print('size of h')
print(len(h))
g = h[:len(h)-1] + [2]
print(g)
print('size of g')
print(len(g))
print('last element of g')
print(g[len(g)-1])