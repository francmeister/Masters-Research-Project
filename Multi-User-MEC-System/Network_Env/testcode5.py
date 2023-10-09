

import numpy as np
import gym
import Network_Env
import pybullet_envs
import torch
import matplotlib.pyplot as plt
import pygame, sys, time, random
import pandas as pd

from numpy import interp

env = gym.make('NetworkEnv-v0')

#timesteps = 5
timesteps = np.arange(0,10,1)
rewards = []
offload_decisions = []
RB_allocations = []
power_allocations = []
throughputs = []
local_delays = []
transmit_delays = []
total_delays = []
energies = []

#state space
channel_gains = []
queue_sizes = []
latencies = []
achieved_channel_rates = []

local_energies = []
transmit_energies = []
reward_ = 0
fiarness_index = []
battery_energies = []
energies_harvested = []
energy_consumed = []

env.reset()
for timestep in timesteps:
    action = env.action_space.sample()
    #print('action: ', action)
    next_observation,reward,dones,info = env.step(action)
    #print(next_observation)
    channel_gains.append(next_observation[0][4])
    achieved_channel_rates.append(env.eMBB_UE_1.achieved_total_energy_consumption)
    
    
    throughputs.append(env.total_rate)
    energies.append(env.total_energy)
    fiarness_index.append(env.SBS1.fairness_index)
    battery_energies.append(env.eMBB_UE_1.battery_energy_level)
    energies_harvested.append(env.eMBB_UE_1.energy_harvested)
    energy_consumed.append(env.eMBB_UE_1.achieved_total_energy_consumption)
    #print('action: ', action)
    #print('reward: ', reward)
    rewards.append(reward[0])
    #print(env.subcarriers)

#achieved_channel_rates.pop(0)
#channel_gains.pop(0)

achieved_channel_rates = np.roll(achieved_channel_rates,-1)
achieved_channel_rates = np.delete(achieved_channel_rates,len(achieved_channel_rates)-1)
channel_gains = np.delete(channel_gains,len(channel_gains)-1)
#achieved_channel_rates[0] = 0
gains_throughputs = {
    'local frequencies': channel_gains,
    'energies': achieved_channel_rates
}

df = pd.DataFrame(data=gains_throughputs)
#print(df)

corr = df.corr(method='pearson')
#print(corr)




    
