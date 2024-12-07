import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


TD3_energy_multiple_users = [0.067788,0.161190,0.254607]
DDPG_energy_multiple_users = []
full_offloading_multiple_users_energy = [0.00016913605774023597,0.0001703231941821021,0.000172211157922183]#3,7,11 users
local_computing_multiple_users_energy = [0.19027157569099212,0.44367963230784163,0.6970266853942844]

TD3_delay_multiple_users = [3.198654,7.425685,11.754907]
DDPG_delay_multiple_users = []
full_offloading_multiple_users_delay = [3.0001200707379643,7.0099352994397455,11.180665901630894]
local_computing_multiple_users_delay = [315.4792111021463,735.0747981843205,1158.8974912077504]

TD3_throughput_multiple_users = [36181414.966455,31511069.986704,34897126.303756]
DDPG_throughput_multiple_users = []
full_offloading_multiple_users_throughput = [33978835.375478804,31914810.185659975,28852512.997813337] # with random resource allocation and power allocation

TD3_fairness_index_multiple_users = [0.710785,0.654622,0.525979]
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