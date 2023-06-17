import numpy as np


max_capacity = 4000

lead_time = 5
obs_dim = lead_time + 4 # total 9

obs_low = np.zeros(obs_dim)
max_mean_daily_demand = 200
max_unit_selling_price = 100
max_daily_holding = 5

obs_high = np.array([max_capacity for _ in range(lead_time)] +
                    [max_mean_daily_demand, max_unit_selling_price,
                     max_daily_holding]
                    )
max_offload_decision = 1
min_offload_decision = 0

number_of_eMBB_users = 7

num_allocate_subcarriers_upper_bound = 25 # get this from the communication channel class
num_allocate_subcarriers_lower_bound = 15 # get this from the communication channel class

max_transmit_power_db = 20
min_transmit_power_db = 1

# my action space consists of
'''
1. offloading decisions for embb user with a maximum of 1 per user
2. number of subcarriers/RB to assign. Range is between self.num_allocate_subcarriers_lower_bound and self.num_allocate_subcarriers_upper_bound Communication channel class
3. uplink transmit power for each embb user with a maximum that is set in the User Equipment class
4. Number of URLLC users per RB 

'''
action_space_high = np.array([max_offload_decision for _ in range(number_of_eMBB_users)] + [num_allocate_subcarriers_upper_bound for _ in range(number_of_eMBB_users)] + 
                        [max_transmit_power_db for _ in range(number_of_eMBB_users)])

action_space_low = np.array([min_offload_decision for _ in range(number_of_eMBB_users)] + [num_allocate_subcarriers_lower_bound for _ in range(number_of_eMBB_users)] + 
                        [min_transmit_power_db for _ in range(number_of_eMBB_users)])
print(obs_low)
print(obs_high)
print("Action Space High: ", action_space_high)
print("Action Space Low: ", action_space_low)

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
g = [1,2,3]
r = [4,4,5]
t = g + r
print(t)