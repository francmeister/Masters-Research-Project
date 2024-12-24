# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import interp


# TD3_energy_multiple_users_20_mhz = [0.061791]
# TD3_energy_multiple_users_40_mhz = [0.254607]
# TD3_energy_multiple_users_60_mhz = [0.573932]
# DDPG_energy_multiple_users = []
# full_offloading_multiple_users_energy_20_mhz = [0.00016858625054827568]#3,7,11 users
# full_offloading_multiple_users_energy_40_mhz = [0.00016766121518017006]#3,7,11 users
# full_offloading_multiple_users_energy_60_mhz = [0.00016711995257541476]#3,7,11 users
# local_computing_multiple_users_energy_20_mhz = [0.08727846453449922]
# local_computing_multiple_users_energy_40_mhz = [0.6970237647718855]
# local_computing_multiple_users_energy_60_mhz = [2.351248012171769]

# TD3_delay_multiple_users_20_mhz = [31.495322]
# TD3_delay_multiple_users_40_mhz  = [11.754907]
# TD3_delay_multiple_users_60_mhz = [11.129204]
# DDPG_delay_multiple_users = []
# full_offloading_multiple_users_delay_20_mhz = [11.234563111762428]
# full_offloading_multiple_users_delay_40_mhz  = [11.13194221278935]
# full_offloading_multiple_users_delay_60_mhz = [11.130206652085006]
# local_computing_multiple_users_delay_20_mhz = [2877.5946902951314]
# local_computing_multiple_users_delay_40_mhz  = [1162.9781233159315]
# local_computing_multiple_users_delay_60_mhz = [585.3011900806567]

# TD3_reward_multiple_users_20_mhz = [8959526085.714914]
# TD3_reward_multiple_users_40_mhz  = [6943804067.788673]
# TD3_reward_multiple_users_60_mhz = [3748722375.906711]

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

# Data
frequencies = ["20 MHz", "40 MHz", "60 MHz"]
td3_energy = [0.061995, 0.254607, 0.573932]
local_energy = [0.08712053071709439, 0.6968702825475757, 2.3503788743343486]
offload_energy = [0.0008616831683168394, 0.0008687623762376314, 0.0008672277227722846]

td3_delay = [54.767543, 46.562085, 46.125478]
local_delay = [381.42673267326734, 371.5818481848186, 288.9918316831679]
offload_delay = [80.57803242653766,83.4328370381999,82.34608374687323]

x = np.arange(len(frequencies))
width = 0.2
# Combined plot with four subplots in the same figure

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# TD3 vs Local Computing (Energy)
axes[0, 0].bar(x - width / 2, td3_energy, width, label='TD3', color='blue')
axes[0, 0].bar(x + width / 2, local_energy, width, label='Local Computing', color='orange')
axes[0, 0].set_title("Energy Consumption: TD3 vs Local Computing")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(frequencies)
axes[0, 0].set_ylabel("Energy (J)")
axes[0, 0].legend()
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# TD3 vs Local Computing (Delay)
axes[0, 1].bar(x - width / 2, td3_delay, width, label='TD3', color='blue')
axes[0, 1].bar(x + width / 2, local_delay, width, label='Local Computing', color='orange')
axes[0, 1].set_title("Delay: TD3 vs Local Computing")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(frequencies)
axes[0, 1].set_ylabel("Delay (ms)")
axes[0, 1].legend()
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

# TD3 vs Full Offloading (Energy)
axes[1, 0].bar(x - width / 2, td3_energy, width, label='TD3', color='blue')
axes[1, 0].bar(x + width / 2, offload_energy, width, label='Full Offloading', color='green')
axes[1, 0].set_title("Energy Consumption: TD3 vs Full Offloading")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(frequencies)
axes[1, 0].set_ylabel("Energy (J)")
axes[1, 0].legend()
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

# TD3 vs Full Offloading (Delay)
axes[1, 1].bar(x - width / 2, td3_delay, width, label='TD3', color='blue')
axes[1, 1].bar(x + width / 2, offload_delay, width, label='Full Offloading', color='green')
axes[1, 1].set_title("Delay: TD3 vs Full Offloading")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(frequencies)
axes[1, 1].set_ylabel("Delay (ms)")
axes[1, 1].legend()
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()


# Observations
# Energy Efficiency:

# Full Offloading consumes the least energy in all scenarios.
# Local Computing is the least efficient, with significant energy consumption increases at higher frequencies.
# Delay:

# Full Offloading and TD3 perform similarly in terms of delay, with TD3 slightly higher in some cases.
# Local Computing has significantly higher delay, especially at lower frequencies.
# Reward:

# TD3 achieves very high rewards, which decrease as the frequency increases.
# Further Analysis
# Why is Full Offloading so energy-efficient?
# It might offload computational tasks to more energy-efficient external systems.
# Trade-off in TD3:
# While energy and delay are higher than Full Offloading, the reward is maximized, indicating better overall system optimization.
# Local Computing's Limitations:
# The method struggles with both energy and delay, especially at higher loads and frequencies.
