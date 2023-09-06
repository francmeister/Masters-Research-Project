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
timesteps = np.arange(0,2,1)
energies = []
offload_decisions = []
RB_allocations = []
power_allocations = []
throughputs = []

#state space
channel_gains = []
queue_sizes = []
latencies = []

local_energies = []
transmit_energies = []

env.reset()
for timestep in timesteps:
    action = env.action_space.sample()
    
    observation,reward,dones,info = env.step(action)
    channel_gains.append(observation[0][0])
    queue_sizes.append(observation[0][1])
    latencies.append(observation[0][2])
    local_energies.append(env.eMBB_UE_1.achieved_local_energy_consumption)
    transmit_energies.append(env.eMBB_UE_1.achieved_transmission_energy_consumption)
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
#print(transmit_energies)
#plt.scatter(timesteps, local_energies, color ="red")
plt.scatter(timesteps,transmit_energies,color = "blue")
#plt.scatter(timesteps,latencies,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
plt.legend(["Local energies", "transmit energies", "latencies"])
plt.xlabel("timesteps")
plt.ylabel("channel gains, queue sizes, latencies")
#plt.title("Throughput vs Number of Allocated RBs")
#plt.show()




    
