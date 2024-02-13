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
timesteps = np.arange(0,30,1)
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
battery_energies = []
energies_harvested = []
energy_consumed = []
local_energy = []
tasks_dropped = []


obs = env.reset()
print('obs')
print(obs)
#print('observation sample')
#print(env.observation_space.sample())
#expl_noise = 0.5
for timestep in timesteps:
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    action = env.action_space.sample()
    action = env.enforce_constraint(action)
    #print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    #print(action)
    #print('')
    #print(action)
    
    #print(timestep)
    #print('action before adding noise')
    #print(action)
    #action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape)).clip(env.action_space.low, env.action_space.high)
    #print('action after adding noise')
    #print(action)
    #print(' ')
    #print('action: ', action)
    observation,reward,dones,info = env.step(action)
    
    #print(observation)
    throughputs.append(env.eMBB_UE_1.achieved_channel_rate_normalized)
    energies.append(env.total_energy)
    fiarness_index.append(env.SBS1.fairness_index)
    battery_energies.append(env.eMBB_UE_1.battery_energy_level)
    energies_harvested.append(env.eMBB_UE_1.energy_harvested)
    energy_consumed.append(env.eMBB_UE_1.achieved_total_energy_consumption)
    channel_gains.append(sum(env.eMBB_UE_1.total_gain[0]))
    power_allocations.append(env.eMBB_UE_1.assigned_transmit_power_W)
    local_energies.append(env.eMBB_UE_1.achieved_local_energy_consumption)
    transmit_energies.append(env.eMBB_UE_1.achieved_transmission_energy_consumption)
    #print('action: ', action)
    #print('reward: ', reward)
    rewards.append(reward[0])
    tasks_dropped.append(env.SBS1.tasks_dropped)
    
  

    #print(sum(reward))
    #print(env.subcarriers)


    #throughputs.append(reward[0])

throughputs = np.roll(throughputs,-1)
power_allocations = np.roll(power_allocations,-1)
data = {
    'channel_gains':channel_gains,
    'transmit_powers':power_allocations,
    'achieved_throughputs':throughputs
}
df = pd.DataFrame(data=data)
print(df)
corr = df.corr(method='pearson')
print(corr)
print('max reward: ', max(rewards), 'min reward: ', min(rewards))
#print(energy_consumed)
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
plt.plot(timesteps, rewards, color ="red")
#plt.plot(timesteps,throughputs, color = "blue")
#plt.scatter(timesteps,transmit_energies,color = "green")
#plt.scatter(timesteps,latencies,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
#plt.legend(["rewards", "transmit energies", "latencies"])
#plt.xlabel("timesteps")
#plt.ylabel("battery energies")
#plt.title("Throughput vs Number of Allocated RBs")
#figure, axis = plt.subplots(1,1)

#axis[0].plot(timesteps, energy_consumed)
#axis[0].set_title('energy consumed')
'''
axis[1].plot(timesteps, energy_consumed)
axis[1].set_title('energy consumed')

axis[2].plot(timesteps, battery_energies)
axis[2].set_title('battery energies')

axis[3].plot(timesteps, rewards)
axis[3].set_title('battery energies reward')
'''


#axis[2].plot(timesteps, rewards)
#axis[2].set_title('rewards')

plt.show()
#plt.show()




    
