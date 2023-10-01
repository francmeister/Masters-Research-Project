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
timesteps = np.arange(0,2000,1)
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

local_energies = []
transmit_energies = []
reward_ = 0
fiarness_index = []

env.reset()
for timestep in timesteps:
    action = env.action_space.sample()
    #print('action: ', action)
    observation,reward,dones,info = env.step(action)
    throughputs.append(env.total_rate)
    energies.append(env.total_energy)
    fiarness_index.append(env.SBS1.fairness_index)
    #print('action: ', action)
    #print('reward: ', reward)
    rewards.append(reward[0])
    #print(env.subcarriers)

data = {
    "Throughputs": throughputs,
    "Energies": energies,
    "Fairness Index": fiarness_index,
    "Total Rewards": rewards
}
    
df = pd.DataFrame(data)

print(df)
x = df.corr(method='pearson')
print(x)
    #throughputs.append(reward[0])
#print('max local delay: ', max(rewards), 'min local delay: ', min(rewards))
#print(rewards)
#print('total reward after 100 timesteps: ', reward_)
#print('offloading decisions: ', env.selected_offload_decisions)
#print('Power allocations: ', env.selected_powers)
#print('RB allocations: ', env.selected_RBs)
#print('Throughputs: ', throughputs)
#print('energies consumed: ', energies)
#print('channel gains: ', channel_gains)
#print('queue sizes: ', queue_sizes)
#print('latencies: ', latencies)
#print('Max Throughput: ', max(throughputs), 'Min Throughput: ', min(throughputs))
#print(transmit_energies)
plt.scatter(timesteps, rewards, color ="red")
#plt.scatter(timesteps,local_delays,color = "blue")
#plt.scatter(timesteps,latencies,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
#plt.legend(["rewards", "transmit energies", "latencies"])
plt.xlabel("timesteps")
plt.ylabel("rewards")
#plt.title("Throughput vs Number of Allocated RBs")
#plt.show()




    
