import numpy as np
import gym
import Network_Env
import pybullet_envs
import torch
import matplotlib.pyplot as plt
import pygame, sys, time, random

from numpy import interp

env = gym.make('NetworkEnv-v0')

#timesteps = 5
timesteps = np.arange(0,10,1)
energies = []
offload_decisions = []
RB_allocations = []
power_allocations = []
throughputs = []

#state space
channel_gains = []
queue_sizes = []
latencies = []

env.reset()
for timestep in timesteps:
    action = env.action_space.sample()
    
    observation,reward,dones,info = env.step(action)
    channel_gains.append(observation[0][0])
    queue_sizes.append(observation[0][1])
    latencies.append(observation[0][2])
    #print('selected actions: ', env.selected_actions)
    #energy_consumed = env.eMBB_Users[0].achieved_total_energy_consumption
    #energies.append(reward[0])
    #throughputs.append(reward[0])

#print('offloading decisions: ', env.selected_offload_decisions)
#print('Power allocations: ', env.selected_powers)
#print('RB allocations: ', env.selected_RBs)
#print('Throughputs: ', throughputs)
#print('energies consumed: ', energies)
#print('channel gains: ', channel_gains)
#print('queue sizes: ', queue_sizes)
#print('latencies: ', latencies)
#print('Max Throughput: ', max(throughputs), 'Min Throughput: ', min(throughputs))
plt.scatter(timesteps, channel_gains, color ="red")
plt.scatter(timesteps,queue_sizes,color = "blue")
plt.scatter(timesteps,latencies,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
plt.legend(["channel gains", "queue sizes", "latencies"])
plt.xlabel("timesteps")
plt.ylabel("channel gains, queue sizes, latencies")
#plt.title("Throughput vs Number of Allocated RBs")
plt.show()




    
