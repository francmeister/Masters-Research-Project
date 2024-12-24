# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import interp


# TD3_energy_multiple_users_type_1 = [0.211393]
# TD3_energy_multiple_users_type_2 = [0.017973]
# TD3_energy_multiple_users_type_3 = [0.020441]
# TD3_energy_multiple_users_type_4 = [0.001014]
# DDPG_energy_multiple_users = []
# full_offloading_multiple_users_energy_type_1 = [0.0001685293140755837]#3,7,11 users
# full_offloading_multiple_users_energy_type_2 = [0.00016864831330618704]#3,7,11 users
# full_offloading_multiple_users_energy_type_3 = [0.00016949044389031852]
# full_offloading_multiple_users_energy_type_4 = [0.00017052497605641608]#3,7,11 users
# local_computing_multiple_users_energy_type_1 = [0.6970100705150036]
# local_computing_multiple_users_energy_type_2 = [0.019562666682258506]
# local_computing_multiple_users_energy_type_3 = [0.17643995683736277]
# local_computing_multiple_users_energy_type_4 = [0.00788278921588087]

# TD3_delay_multiple_users_type_1 = [11.269698]
# TD3_delay_multiple_users_type_2  = [331.243528]
# TD3_delay_multiple_users_type_3 = [11.000582]
# TD3_delay_multiple_users_type_4 = [11.000582]
# DDPG_delay_multiple_users = []
# full_offloading_multiple_users_delay_type_1 = [11.13430555707121]
# full_offloading_multiple_users_delay_type_2  = [11.151867970605862]
# full_offloading_multiple_users_delay_type_3 = [11.000624421136479]
# full_offloading_multiple_users_delay_type_4 = [11.000629801767891]
# local_computing_multiple_users_delay_type_1 = [880.5434067607023]
# local_computing_multiple_users_delay_type_2  = [4690.532855966992]
# local_computing_multiple_users_delay_type_3 = [11.0]
# local_computing_multiple_users_delay_type_4 = [12.669311027674155]

# TD3_reward_multiple_users_type_1 = [7342440973.605168]
# TD3_reward_multiple_users_type_2  = [5972171054.558116]
# TD3_reward_multiple_users_type_3 = [9233405106.570135]
# TD3_reward_multiple_users_type_4 = [9429020876.114311]

# TD3_throughput_multiple_users = []
# DDPG_throughput_multiple_users = []
# full_offloading_multiple_users_throughput = [33978835.375478804,31914810.185659975,28852512.997813337] # with random resource allocation and power allocation

# TD3_fairness_index_multiple_users_10_6 = []
# #TD3_fairness_index_multiple_users_20_6 = [0.710785,0.654622,0.525979]
# DDPG_fairness_index_multiple_users = []
# random_resource_allocation_fairness_index_multiple_users = [0.8534970089224784,0.6627020041731907,0.5150579831673366]

# num_users = [3,7,11]


# figure, axis = plt.subplots(2,2)

# axis[0,0].plot(num_users,TD3_energy_multiple_users, color="green", label=r"TD3", marker='s')
# axis[0,0].plot(num_users,full_offloading_multiple_users_energy, color="red", label=r"Full Offloading",marker='s')
# axis[0,0].plot(num_users,local_computing_multiple_users_energy, color="brown", label=r"Full Local Computing",marker='s')
# #axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
# #axis[0,0].set_title('')
# axis[0,0].grid()
# axis[0,0].set_xlabel('Number of Users')
# axis[0,0].set_ylabel('Energy Consumption (J)')
# axis[0,0].legend(loc="upper left")

# axis[0,1].plot(num_users,TD3_delay_multiple_users, color="green", label=r"TD3", marker='s')
# axis[0,1].plot(num_users,full_offloading_multiple_users_delay, color="red", label=r"Full Offloading",marker='s')
# #axis[0,1].plot(num_users,local_computing_multiple_users_delay, color="brown", label=r"Full Local Computing",marker='s')
# #axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
# #axis[0,0].set_title('')
# axis[0,1].grid()
# axis[0,1].set_xlabel('Number of Users')
# axis[0,1].set_ylabel('Task Delay (ms)')
# axis[0,1].legend(loc="upper left")

# axis[1,0].plot(num_users,TD3_delay_multiple_users, color="green", label=r"TD3", marker='s')
# #axis[1,0].plot(num_users,full_offloading_multiple_users_delay, color="red", label=r"Full Offloading",marker='s')
# axis[1,0].plot(num_users,local_computing_multiple_users_delay, color="brown", label=r"Full Local Computing",marker='s')
# #axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
# #axis[0,0].set_title('')
# axis[1,0].grid()
# axis[1,0].set_xlabel('Number of Users')
# axis[1,0].set_ylabel('Task Delay (ms)')
# axis[1,0].legend(loc="upper left")


# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data for energy consumption (all values are lists with single elements)
user_types = ['Type 1', 'Type 2', 'Type 3', 'Type 4']

task_sizes = ['50 bits/task', '500 bits/task', '1000 bits/task']

TD3_energy = [0.020642, 0.211591, 0.416773]
full_offloading_energy = [0.0008600990099009983, 0.0008649504950495118, 0.000870049504950502]
local_computing_energy = [0.17663618515910118,0.6966616493624364, 0.6969728707135012]

# Data for delay
TD3_delay = [8.895596, 25.541444, 143.752507]
full_offloading_delay = [9.11727291287328, 29.68475815866853, 146.690621812883]
local_computing_delay = [10.814851485148514,343.96287128712873, 439.91980198019803]

x = np.arange(len(task_sizes))  # label locations
width = 0.25  # width of the bars
# Plot for Energy and Delay Comparisons (4 Subplots)
fig, ax = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: TD3 vs Local Computing (Energy)
ax[0, 0].bar(x - width/2, TD3_energy, width, label='TD3', color='royalblue')
ax[0, 0].bar(x + width/2, local_computing_energy, width, label='Local Computing', color='firebrick')

ax[0, 0].set_xlabel('User Types')
ax[0, 0].set_ylabel('Energy Consumption (J)')
ax[0, 0].set_title('Energy: TD3 vs Local Computing')
ax[0, 0].set_xticks(x)
ax[0, 0].set_xticklabels(task_sizes)
ax[0, 0].legend()
ax[0, 0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: TD3 vs Local Computing (Delay)
ax[0, 1].bar(x - width/2, TD3_delay, width, label='TD3', color='royalblue')
ax[0, 1].bar(x + width/2, local_computing_delay, width, label='Local Computing', color='firebrick')

ax[0, 1].set_xlabel('User Types')
ax[0, 1].set_ylabel('Delay (ms)')
ax[0, 1].set_title('Delay: TD3 vs Local Computing')
ax[0, 1].set_xticks(x)
ax[0, 1].set_xticklabels(task_sizes)
ax[0, 1].legend()
ax[0, 1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: TD3 vs Full Offloading (Energy)
ax[1, 0].bar(x - width/2, TD3_energy, width, label='TD3', color='royalblue')
ax[1, 0].bar(x + width/2, full_offloading_energy, width, label='Full Offloading', color='seagreen')

ax[1, 0].set_xlabel('User Types')
ax[1, 0].set_ylabel('Energy Consumption (J)')
ax[1, 0].set_title('Energy: TD3 vs Full Offloading')
ax[1, 0].set_xticks(x)
ax[1, 0].set_xticklabels(task_sizes)
ax[1, 0].legend()
ax[1, 0].grid(True, linestyle='--', alpha=0.6)

# Plot 4: TD3 vs Full Offloading (Delay)
ax[1, 1].bar(x - width/2, TD3_delay, width, label='TD3', color='royalblue')
ax[1, 1].bar(x + width/2, full_offloading_delay, width, label='Full Offloading', color='seagreen')

ax[1, 1].set_xlabel('User Types')
ax[1, 1].set_ylabel('Delay (ms)')
ax[1, 1].set_title('Delay: TD3 vs Full Offloading')
ax[1, 1].set_xticks(x)
ax[1, 1].set_xticklabels(task_sizes)
ax[1, 1].legend()
ax[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Task Characteristics
# Type 1: Large task size (500 bits/slot) with high computational demand (330 cycles/bit).
# Type 2: Large task size (500 bits/slot) but lower computational demand (100 cycles/bit).
# Type 3: Small task size (50 bits/slot) but high computational demand (330 cycles/bit).
# Type 4: Small task size (50 bits/slot) with low computational demand (100 cycles/bit).
# These characteristics play a significant role in the energy and delay performance of TD3, Local Computing, and Full Offloading.

# Updated Deductions
# 1. Energy Consumption
# Type 1 (High size, high computation)

# Local computing consumes the most energy due to the large task size and high computational demand.
# TD3 reduces energy consumption significantly (by 69.67%) compared to local computing but still requires more energy than Full Offloading.
# Full Offloading is extremely efficient, reducing energy by 99.92% relative to TD3, because computation is done on a remote server with less energy cost on the local device.
# Type 2 (High size, low computation)

# Since computational demand is lower (100 cycles/bit), local computing consumes less energy compared to Type 1.
# However, Full Offloading still performs best, offering a 99.06% reduction in energy compared to TD3.
# TD3 provides only a 8.13% reduction in energy relative to local computing, suggesting that TD3's benefits are limited for tasks with low computation requirements.
# Type 3 (Small size, high computation)

# The task size is smaller, but the high computational demand (330 cycles/bit) still makes local computing energy-intensive.
# TD3 achieves a large 88.41% reduction in energy relative to local computing.
# Full Offloading once again achieves an energy reduction of 99.17% relative to TD3, making it the most efficient for this type of task.
# Type 4 (Small size, low computation)

# Small task size and low computational requirements result in the lowest energy consumption for all strategies.
# Local computing still consumes more energy than TD3, but the reduction is about 87.14% — a significant savings.
# Full Offloading, as expected, offers an 83.18% reduction in energy compared to TD3, but the absolute values are already very small.
# 2. Delay
# Type 1 (High size, high computation)

# Local computing has a very high delay (880.54 ms) due to the large task size and high computation demand.
# TD3 reduces delay by 98.72%, bringing it down to 11.27 ms, which is close to Full Offloading’s delay (11.13 ms).
# Full Offloading performs slightly better than TD3, achieving an additional 1.20% reduction in delay.
# Type 2 (High size, low computation)

# Local computing suffers from the highest delay (4690.53 ms) because of the large task size.
# TD3 reduces delay by 92.94%, bringing it down to 331.24 ms.
# Full Offloading provides a significant delay improvement over TD3, achieving a 96.63% reduction.
# The large task size increases transmission time in Full Offloading, but it still performs better than TD3.
# Type 3 (Small size, high computation)

# Since the task size is small, Local Computing can achieve low delays (11.0 ms).
# TD3 has the same delay (11.00 ms), meaning the computational load doesn't influence delay much in this case.
# Full Offloading offers no significant improvement here. The delays for TD3 and Full Offloading are nearly identical.
# Type 4 (Small size, low computation)

# Local computing has a delay of 12.67 ms, which is still higher than the 11.00 ms for TD3.
# TD3 reduces delay by 13.17% relative to local computing.
# Full Offloading shows no significant improvement over TD3, as both have almost identical delays.
# Updated Key Takeaways
# Energy Perspective
# For large task sizes (Type 1 and Type 2), Full Offloading is the best approach since it provides over 99% energy reduction.
# For small task sizes (Type 3 and Type 4), TD3 provides substantial energy savings relative to Local Computing, but Full Offloading remains the best option.
# If energy efficiency is a top priority, Full Offloading is the clear choice for all task types.
# Delay Perspective
# For tasks with large sizes (Type 1 and Type 2), Full Offloading performs slightly better than TD3, especially for Type 2.
# For tasks with small sizes (Type 3 and Type 4), TD3 and Full Offloading have nearly identical delays.
# Local Computing has much higher delays for large task sizes (Type 1 and Type 2), making it a poor choice for delay-sensitive applications.