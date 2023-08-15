import numpy as np
import gym
import Network_Env
import pybullet_envs
import torch

from numpy import interp
print("Interpolated number: ",interp(0.8333,[0,1],[0,25]))

env = gym.make('NetworkEnv-v0')
#env = gym.make('AntBulletEnv-v0')
env.reset()
max_capacity = 4000
action = env.action_space.sample()
observation = env.observation_space.sample()
print("Observation Space sample:")
print(observation)
print('observation space high')
print(env.observation_space.high)
print('observation space low')
print(env.observation_space.low)
print("max_action:")
print(env.action_space.high)
print("min_action:")
print(env.action_space.low)
print("Action Sample before transpose")
print(action)
#observation,reward,done,info = env.step(action)
print("observation space dim:")
print(env.observation_space.shape)
print("action space dim:")
print(env.action_space.shape)
env.step(action)
dones = [1,2,3,4,5]
print(dones[5])
print(len(dones))

# my action space consists of
'''
1. offloading decisions for embb user with a maximum of 1 per user
2. number of subcarriers/RB to assign. Range is between self.num_allocate_subcarriers_lower_bound and self.num_allocate_subcarriers_upper_bound Communication channel class
3. uplink transmit power for each embb user with a maximum that is set in the User Equipment class
4. Number of URLLC users per RB 

'''
'''
action_space = Box(low=np.array([min_offload_decision,num_allocate_subcarriers_lower_bound,min_transmit_power_db,min_number_of_URLLC_users_per_RB]),
                   high=np.array([max_offload_decision,num_allocate_subcarriers_upper_bound,max_transmit_power_db,max_number_of_URLLC_users_per_RB]),
                   shape=(4,number_of_eMBB_users),dtype=np.float32)

observation_space = Box(low=np.array([channel_gain_min,communication_queue_min,energy_harvested_min,latency_requirement_min,reliability_requirement_min]),
                        high=np.array([channel_gain_max,communication_queue_max,energy_harvested_max,latency_requirement_max,reliability_requirement_max]),
                        shape=(5,number_of_users),dtype=np.float32)

action_space_high = np.array([max_offload_decision for _ in range(number_of_eMBB_users)] + [num_allocate_subcarriers_upper_bound for _ in range(number_of_eMBB_users)] + 
                        [max_transmit_power_db for _ in range(number_of_eMBB_users)])

action_space_low = np.array([min_offload_decision for _ in range(number_of_eMBB_users)] + [num_allocate_subcarriers_lower_bound for _ in range(number_of_eMBB_users)] + 
                        [min_transmit_power_db for _ in range(number_of_eMBB_users)])
print(obs_low)
print(obs_high)
print("Action Space High: ", action_space_high)
print("Action Space Low: ", action_space_low)
'''


#################################################################################################################################

#Observation Space

#channel_gain,communication_queue,energy_harvested, QOS_requirements

'''
1. Channel gain of each user, start with URLLC users. Find lower and upper limits
2. Size of communication queue of each user. Start with URLLC users. Find lower and upper limits
3. Energy harvested by each user. Start with URLLC users. Find lower and upper limits.
4. QOS requirement - latency requirement and reliability requirement
'''
#channel gain


