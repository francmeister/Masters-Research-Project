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

TD3_fairness_index_multiple_users_10_6 = [0.710785,0.654622,0.525979]
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




energy_constant_values = [10**(-15), 10**(-18) , 10**(-20), 10**(-25)]
#energy_constant_values = np.log10(energy_constant_values)

#-------------------------------------Model trained with 40 MHz-------------------------------------
TD3_11_users_energy_consumption = [0.254799, 0.000910,0.000658,0.000656]
full_offloading_11_users_energy_consumption = [0.0008680693069307004,0.00087797029702971,0.0008680693069307004,0.0008680693069307004]
full_local_computing_11_users_energy_consumption = [0.6968477472133169,0.000696927813996256,6.96908042078842e-06,6.969839291191688e-11]

TD3_11_users_delay = [41.154899,46.572479,46.574979,46.560401]
full_offloading_11_users_delay = [80.57803242653766, 80.57803242653766,83.4328370381999,82.34608374687323]
full_local_computing_11_users_delay = [381.26419141914175,379.5909240924093,378.80330033003315,379.4546204620462]

#-------------------------------------Model trained with 110 MHz-------------------------------------
TD3_11_users_energy_consumption = [1.931876, 0.002587,0.000675,0.000656]
full_offloading_11_users_energy_consumption = [0.0008680693069307004,0.00087797029702971,0.0008680693069307004,0.0008680693069307004]
full_local_computing_11_users_energy_consumption = [14.369465130305837,0.014393208459102622,0.00014391352639153224,1.4366189850960628e-09]

TD3_11_users_delay = [45.981812,45.982021,45.975727,45.993555]
full_offloading_11_users_delay = [80.57803242653766, 80.57803242653766,83.4328370381999,82.34608374687323]
full_local_computing_11_users_delay = [75.30400990099005,74.2597029702971,79.11912541254124,74.13587458745876]
# Adjusted Bar Graph Visualization for Zoomed-In Energy Consumption
fig, axs = plt.subplots(2, 4, figsize=(18, 10))

# Titles for each subplot
titles = ["Energy Constant Value = 10^(-15)", 
          "Energy Constant Value = 10^(-18)",
          "Energy Constant Value = 10^(-20)", 
          "Energy Constant Value = 10^(-25)"]

# Energy data and delay data
energy_data = [TD3_11_users_energy_consumption, 
               full_offloading_11_users_energy_consumption, 
               full_local_computing_11_users_energy_consumption]

delay_data = [TD3_11_users_delay, 
              full_offloading_11_users_delay, 
              full_local_computing_11_users_delay]

labels = ['TD3', 'Full Offloading', 'Full Local Computing']

# Bar plot for energy consumption
for i, energy in enumerate(zip(*energy_data)):
    axs[0, i].bar(labels, energy, color=['blue', 'orange', 'green'])
    axs[0, i].set_title(f"Energy Consumption\n{titles[i]}")
    axs[0, i].set_ylabel("Energy Consumption")
    if i == 0:
        axs[0, i].set_ylim(0, max(max(energy_data)) * 1.2)  # Default for 10^(-15)
    else:
        axs[0, i].set_ylim(0, 0.001)  # Zoomed-in for 10^(-20) and 10^(-25)
    axs[0, i].grid(axis='y')

# Bar plot for delay
for i, delay in enumerate(zip(*delay_data)):
    axs[1, i].bar(labels, delay, color=['blue', 'orange', 'green'])
    axs[1, i].set_title(f"Delay\n{titles[i]}")
    axs[1, i].set_ylabel("Delay")
    axs[1, i].set_ylim(0, max(max(delay_data)) * 1.2)
    axs[1, i].grid(axis='y')

plt.tight_layout()
plt.show()
