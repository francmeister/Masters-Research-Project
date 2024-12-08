import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


TD3_energy_multiple_users_20_mhz = []
TD3_energy_multiple_users_40_mhz = [0.254607]
TD3_energy_multiple_users_60_mhz = [0.573932]
DDPG_energy_multiple_users = []
full_offloading_multiple_users_energy_20_mhz = []#3,7,11 users
full_offloading_multiple_users_energy_40_mhz = []#3,7,11 users
full_offloading_multiple_users_energy_60_mhz = []#3,7,11 users
local_computing_multiple_users_energy_20_mhz = []
local_computing_multiple_users_energy_40_mhz = []
local_computing_multiple_users_energy_60_mhz = []

TD3_delay_multiple_users_20_mhz = []
TD3_delay_multiple_users_40_mhz  = [11.754907]
TD3_delay_multiple_users_60_mhz = [11.129204]
DDPG_delay_multiple_users = []
full_offloading_multiple_users_delay_20_mhz = []
full_offloading_multiple_users_delay_40_mhz  = []
full_offloading_multiple_users_delay_60_mhz = []
local_computing_multiple_users_delay_20_mhz = []
local_computing_multiple_users_delay_40_mhz  = []
local_computing_multiple_users_delay_60_mhz = []

TD3_reward_multiple_users_20_mhz = []
TD3_reward_multiple_users_40_mhz  = []
TD3_reward_multiple_users_60_mhz = [3748722375.906711]

TD3_throughput_multiple_users = []
DDPG_throughput_multiple_users = []
full_offloading_multiple_users_throughput = [33978835.375478804,31914810.185659975,28852512.997813337] # with random resource allocation and power allocation

TD3_fairness_index_multiple_users_10_6 = []
#TD3_fairness_index_multiple_users_20_6 = [0.710785,0.654622,0.525979]
DDPG_fairness_index_multiple_users = []
random_resource_allocation_fairness_index_multiple_users = [0.8534970089224784,0.6627020041731907,0.5150579831673366]

num_users = [3,7,11]


figure, axis = plt.subplots(2,2)

axis[0,0].plot(num_users,TD3_energy_multiple_users, color="green", label=r"TD3", marker='s')
axis[0,0].plot(num_users,full_offloading_multiple_users_energy, color="red", label=r"Full Offloading",marker='s')
axis[0,0].plot(num_users,local_computing_multiple_users_energy, color="brown", label=r"Full Local Computing",marker='s')
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
#axis[0,0].set_title('')
axis[0,0].grid()
axis[0,0].set_xlabel('Number of Users')
axis[0,0].set_ylabel('Energy Consumption (J)')
axis[0,0].legend(loc="upper left")

axis[0,1].plot(num_users,TD3_delay_multiple_users, color="green", label=r"TD3", marker='s')
axis[0,1].plot(num_users,full_offloading_multiple_users_delay, color="red", label=r"Full Offloading",marker='s')
#axis[0,1].plot(num_users,local_computing_multiple_users_delay, color="brown", label=r"Full Local Computing",marker='s')
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
#axis[0,0].set_title('')
axis[0,1].grid()
axis[0,1].set_xlabel('Number of Users')
axis[0,1].set_ylabel('Task Delay (ms)')
axis[0,1].legend(loc="upper left")

axis[1,0].plot(num_users,TD3_delay_multiple_users, color="green", label=r"TD3", marker='s')
#axis[1,0].plot(num_users,full_offloading_multiple_users_delay, color="red", label=r"Full Offloading",marker='s')
axis[1,0].plot(num_users,local_computing_multiple_users_delay, color="brown", label=r"Full Local Computing",marker='s')
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
#axis[0,0].set_title('')
axis[1,0].grid()
axis[1,0].set_xlabel('Number of Users')
axis[1,0].set_ylabel('Task Delay (ms)')
axis[1,0].legend(loc="upper left")


plt.tight_layout()
plt.show()