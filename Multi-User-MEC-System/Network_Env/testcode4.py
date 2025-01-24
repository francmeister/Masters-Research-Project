import sys
import numpy as np
import gym
import Network_Env
import pybullet_envs
#import torch
import matplotlib.pyplot as plt
#import pygame, sys, time, random
import pandas as pd
from numpy import interp

env = gym.make('NetworkEnv-v0')

timesteps = np.arange(0,env.STEP_LIMIT,1)
num_episodes = 1
episodes = np.arange(0,num_episodes,1)
obs = env.reset()
number_of_users = env.number_of_users
number_of_RBs = env.num_allocate_RB_upper_bound
small_scale_channel_gains = []
large_scale_channel_gains = []
battery_energy_levels = []
local_queue_lengths = []
offloading_queue_lengths = []
#observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths,num_urllc_arriving_packets)) #observation_channel_gains.
for episode in episodes:
    for timestep in timesteps:
        #print('----------------------------------------------------------------------------------------------------------------------------------------------------')
        action = env.action_space.sample()
        action = env.enforce_constraint(action)
        #print(action)
        action2, action = env.reshape_action_space_dict(action)
        observation,reward,dones,info = env.step_(action)
        small_scale_channel_gains.append(observation[0:number_of_users*number_of_RBs])
        large_scale_channel_gains.append(observation[number_of_users*number_of_RBs:(number_of_users*number_of_RBs)*2])
        battery_energy_levels.append(observation[(number_of_users*number_of_RBs)*2:(number_of_users*number_of_RBs)*2+number_of_users])
        offloading_queue_lengths.append(observation[(number_of_users*number_of_RBs)*2+number_of_users:(number_of_users*number_of_RBs)*2+number_of_users*2])
        local_queue_lengths.append(observation[(number_of_users*number_of_RBs)*2+number_of_users*2:(number_of_users*number_of_RBs)*2+number_of_users*3])
        #print('observation:')
        #print(observation)
        #print('----------------------------------------------------------')
        #print('small_scale_channel_gains:')
        #print(observation[0:number_of_users*number_of_RBs])
        #print('----------------------------------------------------------')
        #print('battery_energy_levels')
        #print(observation[(number_of_users*number_of_RBs)*2:(number_of_users*number_of_RBs)*2+number_of_users])
        #print('----------------------------------------------------------')
        #print('large_scale_channel_gains:')
        #print(observation[number_of_users*number_of_RBs:(number_of_users*number_of_RBs)*2])
        #print('----------------------------------------------------------')
        #print('offloading_queue_lengths:')
        #print(observation[(number_of_users*number_of_RBs)*2+number_of_users:(number_of_users*number_of_RBs)*2+number_of_users*2])
        #print('----------------------------------------------------------')
        #print('local_queue_lengths')
        #print(observation[(number_of_users*number_of_RBs)*2+number_of_users*2:(number_of_users*number_of_RBs)*2+number_of_users*3])

small_scale_channel_gains = np.array(small_scale_channel_gains)
small_scale_channel_gains_x_dim = len(small_scale_channel_gains)
small_scale_channel_gains_y_dim = len(small_scale_channel_gains[0])
small_scale_channel_gains = small_scale_channel_gains.reshape(1,small_scale_channel_gains_x_dim*small_scale_channel_gains_y_dim)
small_scale_channel_gains = small_scale_channel_gains.squeeze()

large_scale_channel_gains = np.array(large_scale_channel_gains)
large_scale_channel_gains_x_dim = len(large_scale_channel_gains)
large_scale_channel_gains_y_dim = len(large_scale_channel_gains[0])
large_scale_channel_gains = large_scale_channel_gains.reshape(1,large_scale_channel_gains_x_dim*large_scale_channel_gains_y_dim)
large_scale_channel_gains = large_scale_channel_gains.squeeze()

offloading_queue_lengths = np.array(offloading_queue_lengths)
offloading_queue_lengths_x_dim = len(offloading_queue_lengths)
offloading_queue_lengths_y_dim = len(offloading_queue_lengths[0])
offloading_queue_lengths = offloading_queue_lengths.reshape(1,offloading_queue_lengths_x_dim*offloading_queue_lengths_y_dim)
offloading_queue_lengths = offloading_queue_lengths.squeeze()

local_queue_lengths = np.array(local_queue_lengths)
local_queue_lengths_x_dim = len(local_queue_lengths)
local_queue_lengths_y_dim = len(local_queue_lengths[0])
local_queue_lengths = local_queue_lengths.reshape(1,local_queue_lengths_x_dim*local_queue_lengths_y_dim)
local_queue_lengths = local_queue_lengths.squeeze()

battery_energy_levels = np.array(battery_energy_levels)
battery_energy_levels_x_dim = len(battery_energy_levels)
battery_energy_levels_y_dim = len(battery_energy_levels[0])
battery_energy_levels = battery_energy_levels.reshape(1,battery_energy_levels_x_dim*battery_energy_levels_y_dim)
battery_energy_levels = battery_energy_levels.squeeze()

# print('small_scale_channel_gains:')
# print(small_scale_channel_gains)
# print('----------------------------------------------------------')
# print('large_scale_channel_gains:')
# print(large_scale_channel_gains)
# print('----------------------------------------------------------')
# print('offloading_queue_lengths:')
# print(offloading_queue_lengths)
# print('----------------------------------------------------------')
# print('local_queue_lengths')
# print(local_queue_lengths)
# print('----------------------------------------------------------')
# print('battery_energy_levels')
# print(battery_energy_levels)

max_small_scale_channel_gain = max(small_scale_channel_gains)
min_small_scale_channel_gain = min(small_scale_channel_gains)

max_large_scale_channel_gain = max(large_scale_channel_gains)
min_large_scale_channel_gain = min(large_scale_channel_gains)

min_local_queue_length = 0
min_offloading_queue_length = 0

max_local_queue_length = max(local_queue_lengths)
max_offloading_queue_length = max(offloading_queue_lengths)

max_local_queue_length_ = max(max_local_queue_length,max_offloading_queue_length)
max_offloading_queue_length_ = max(max_local_queue_length,max_offloading_queue_length)

min_battery_energy_level = 0
max_battery_energy_level = max(battery_energy_levels)

#print('max_battery_energy_level: ', max_battery_energy_level)

env.max_small_scale_channel_gain = max_small_scale_channel_gain
env.min_small_scale_channel_gain = min_small_scale_channel_gain

env.max_battery_energy_level = max_battery_energy_level
env.min_battery_energy_level = min_battery_energy_level

env.max_large_scale_channel_gain = max_large_scale_channel_gain
env.min_large_scale_channel_gain = min_large_scale_channel_gain

env.max_local_queue_length = max_local_queue_length_
env.min_local_queue_length = min_local_queue_length

env.max_offloading_queue_length = max_offloading_queue_length_
env.min_offloading_queue_length = min_offloading_queue_length
#print('env.max_battery_energy_level: ', env.max_battery_energy_level )
#timesteps = 5
timesteps = np.arange(0,100,1)
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
delays = []
local_queue_violation_probabilities = []
offload_ratios = []


obs = env.reset()
print('obs')
print(obs)
print('env.observation_space.sample()')
print(env.observation_space.sample())
print('env.action_space.sample()')
print(env.action_space.sample())
#print('observation sample')
#print(env.observation_space.sample())
#expl_noise = 0.5
#print('env.max_battery_energy_level:', env.max_battery_energy_level)
env.change_state_limits(min_small_scale_channel_gain,max_small_scale_channel_gain,
                            min_large_scale_channel_gain,max_large_scale_channel_gain,
                            min_battery_energy_level,max_battery_energy_level,
                            min_local_queue_length,max_local_queue_length,
                            min_offloading_queue_length,max_offloading_queue_length)
for timestep in timesteps:
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
    action = env.action_space.sample()
    action = env.enforce_constraint(action)
    #print(action)
    action2, action = env.reshape_action_space_dict(action)
    #print('')
    #print(action)

    #action = env.reshape_action_space_from_model_to_dict(action)
    # print('action2')
    # print(action)
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
    #print('env.max_battery_energy_level before step:', env.max_battery_energy_level)
    observation,reward,dones,info = env.step(action)
    #print('env.max_battery_energy_level after step:', env.max_battery_energy_level)
    
    #print(observation)
    throughputs.append(env.eMBB_UE_1.achieved_channel_rate_normalized)
    energies.append(env.total_energy)
    fiarness_index.append(env.SBS1.fairness_index)
    battery_energies.append(env.eMBB_UE_1.battery_energy_level)
    energies_harvested.append(env.eMBB_UE_1.energy_harvested)
    energy_consumed.append(env.eMBB_UE_1.achieved_total_energy_consumption)
    channel_gains.append(sum(env.eMBB_UE_1.total_gain_[0]))
    power_allocations.append(env.eMBB_UE_1.assigned_transmit_power_W)
    local_energies.append(env.eMBB_UE_1.achieved_local_energy_consumption)
    transmit_energies.append(env.eMBB_UE_1.achieved_transmission_energy_consumption)
    delays.append(env.SBS1.delays)
    local_queue_violation_probabilities.append(env.eMBB_UE_1.local_queue_delay_violation_probability_)
    offload_ratios.append(env.eMBB_UE_1.allocated_offloading_ratio)
    #print('action: ', action)
    #print('reward: ', reward)
    rewards.append(reward)
    tasks_dropped.append(env.SBS1.tasks_dropped)
    
  

    #print(sum(reward))
    #print(env.subcarriers)


    #throughputs.append(reward[0])

# individual_local_queue_delays
# individual_offload_queue_delays
# individual_local_queue_lengths
# individual_offload_queue_lengths
print('env.SBS1.individual_local_queue_delays')
print(env.SBS1.individual_local_queue_delays)
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
plt.plot(offload_ratios, local_queue_violation_probabilities, color ="red")
#plt.plot(timesteps,throughputs, color = "blue")
#plt.scatter(timesteps,transmit_energies,color = "green")
#plt.scatter(timesteps,latencies,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
#plt.legend(["rewards", "transmit energies", "latencies"])
plt.xlabel("offloading ratio")
plt.ylabel("probability of violation")
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

#plt.show()
#plt.show()




    
