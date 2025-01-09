import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


# energy_constant = [-24,-21,-18,-15,-12]
# throughput_values = [34309197.737593,34309197.737593,34309197.737593,29545422.636385,26865061.027919]
# reward_values =[2857220516.464021,2857169875.280542,2854435544.804395,6358566585.796227,-7033121360536.943359]
# energy_values = [0.000668,0.000669,0.000922,0.254807,696.685543]
# delay_values = [36.643457,36.645600,36.643500,46.564509,354.193430]

import matplotlib.pyplot as plt


#energy_constant = [-24, -21, -18, -15, -12]
energy_constant = [10**(-24), 10**(-21), 10**(-18), 10**(-15), 10**(-12)]
energy_constant = np.log10(energy_constant)
throughput_values = [34309197.737593, 34309197.737593, 28338530.402582, 29545422.636385, 27543488.879054]
reward_values = [-3886084586.480790, -3888629449.588085, 4297368036.677872, 6358566585.796227, 6729199420.597358]
energy_values = [0.000668, 0.000669, 0.000415, 0.254807, 439.185179]
delay_values = [36.643454, 36.641748, 77.997793, 46.564509, 164.962531]

# Create subplots for energy constant vs each variable
fig, axs = plt.subplots(2, 2)

axs = axs.flatten()

# Plot energy constant vs throughput
axs[0].plot(energy_constant, throughput_values, marker='o', label="Throughput")
axs[0].set_title('Throughput vs Energy Constant')
axs[0].set_xlabel('Energy Constant')
axs[0].set_ylabel('Throughput')
axs[0].grid(True)
axs[0].legend()

# Plot energy constant vs reward
axs[1].plot(energy_constant, reward_values, marker='o', color='orange', label="Reward")
axs[1].set_title('Reward vs Energy Constant')
axs[1].set_xlabel('Energy Constant')
axs[1].set_ylabel('Reward')
axs[1].grid(True)
axs[1].legend()

# Plot energy constant vs energy values
axs[2].plot(energy_constant, energy_values, marker='o', color='green', label="Energy")
axs[2].set_title('Energy Consumption vs Energy Constant')
axs[2].set_xlabel('Energy Constant')
axs[2].set_ylabel('Energy')
axs[2].grid(True)
axs[2].legend()

# Plot energy constant vs delay
axs[3].plot(energy_constant, delay_values, marker='o', color='red', label="Delay")
axs[3].set_title('Delay vs Energy Constant')
axs[3].set_xlabel('Energy Constant')
axs[3].set_ylabel('Delay')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()


rewards_throughput_energy_10_12 = np.load('timestep_rewards_energy_throughput_10_12.npy')
rewards_throughput_energy_10_15 = np.load('timestep_rewards_energy_throughput_10_15.npy')
rewards_throughput_energy_10_18 = np.load('timestep_rewards_energy_throughput_10_18.npy')
rewards_throughput_energy_10_21 = np.load('timestep_rewards_energy_throughput_10_21.npy')
rewards_throughput_energy_10_24 = np.load('timestep_rewards_energy_throughput_10_24.npy')

overall_users_reward_10_12 = np.load('overall_users_reward_10_12.npy')
overall_users_reward_10_15 = np.load('overall_users_reward_10_15.npy')
overall_users_reward_10_18 = np.load('overall_users_reward_10_18.npy')
overall_users_reward_10_21 = np.load('overall_users_reward_10_21.npy')
overall_users_reward_10_24 = np.load('overall_users_reward_10_24.npy')

energy_rewards_10_12 = rewards_throughput_energy_10_12[:,2]
energy_rewards_10_15 = rewards_throughput_energy_10_15[:,2]
energy_rewards_10_18 = rewards_throughput_energy_10_18[:,2]
energy_rewards_10_21 = rewards_throughput_energy_10_21[:,2]
energy_rewards_10_24 = rewards_throughput_energy_10_24[:,2]
#energy_rewards_256_steps = rewards_throughput_energy_256_steps[:,2]

throughput_rewards_10_12 = rewards_throughput_energy_10_12[:,3]
throughput_rewards_10_15 = rewards_throughput_energy_10_15[:,3]
throughput_rewards_10_18 = rewards_throughput_energy_10_18[:,3]
throughput_rewards_10_21 = rewards_throughput_energy_10_21[:,3]
throughput_rewards_10_24 = rewards_throughput_energy_10_24[:,3]
#throughput_rewards_256_steps = rewards_throughput_energy_256_steps[:,3]

delay_rewards_10_12 = rewards_throughput_energy_10_12[:,4]
delay_rewards_10_15 = rewards_throughput_energy_10_15[:,4]
delay_rewards_10_18 = rewards_throughput_energy_10_18[:,4]
delay_rewards_10_21 = rewards_throughput_energy_10_18[:,4]
delay_rewards_10_24 = rewards_throughput_energy_10_18[:,4]
#delay_rewards_256_steps = rewards_throughput_energy_256_steps[:,4]
#overall_users_reward_11_users = np.load('overall_users_reward_TD3_11_users.npy')


#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_10_12 = rewards_throughput_energy_10_12[:,0]
timesteps_10_15 = rewards_throughput_energy_10_15[:,0]
timesteps_10_18 = rewards_throughput_energy_10_18[:,0]

def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size =100

overall_users_reward_10_12 = moving_average(overall_users_reward_10_12, window_size)
overall_users_reward_10_15 = moving_average(overall_users_reward_10_15, window_size)
overall_users_reward_10_18 = moving_average(overall_users_reward_10_18, window_size)
overall_users_reward_10_21 = moving_average(overall_users_reward_10_21, window_size)
overall_users_reward_10_24 = moving_average(overall_users_reward_10_24, window_size)

energy_rewards_10_12 = moving_average(energy_rewards_10_12, window_size)
energy_rewards_10_15 = moving_average(energy_rewards_10_15, window_size)
energy_rewards_10_18 = moving_average(energy_rewards_10_18, window_size)
energy_rewards_10_21 = moving_average(energy_rewards_10_21, window_size)
energy_rewards_10_24 = moving_average(energy_rewards_10_24, window_size)

throughput_rewards_10_12 = moving_average(throughput_rewards_10_12, window_size)
throughput_rewards_10_15 = moving_average(throughput_rewards_10_15, window_size)
throughput_rewards_10_18 = moving_average(throughput_rewards_10_18, window_size)
throughput_rewards_10_21 = moving_average(throughput_rewards_10_21, window_size)
throughput_rewards_10_24 = moving_average(throughput_rewards_10_24, window_size)

delay_rewards_10_12 = moving_average(delay_rewards_10_12, window_size)
delay_rewards_10_15 = moving_average(delay_rewards_10_15, window_size)
delay_rewards_10_18 = moving_average(delay_rewards_10_18, window_size)
delay_rewards_10_21 = moving_average(delay_rewards_10_21, window_size)
delay_rewards_10_24 = moving_average(delay_rewards_10_24, window_size)

figure, axis = plt.subplots(2,2)

#axis[0,0].plot(timesteps_10_12[window_size-1:], overall_users_reward_10_12, color="green", label=r"$10^{-12}$")
axis[0,0].plot(timesteps_10_15[window_size-1:], overall_users_reward_10_15, color="red", label=r"$10^{-15}$")
axis[0,0].plot(timesteps_10_15[window_size-1:], overall_users_reward_10_18, color="blue", label=r"$10^{-18}$")
#axis[0,0].plot(timesteps_10_15[window_size-1:], overall_users_reward_10_21, color="blue", label=r"$10^{-21}$")
#axis[0,0].plot(timesteps_10_15[window_size-1:], overall_users_reward_10_24, color="black", label=r"$10^{-24}$")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,0].set_title('Total System Reward')
axis[0,0].grid()
axis[0,0].set_xlabel('Timestep')
axis[0,0].legend(loc="lower right")

#axis[0,1].plot(timesteps_10_12[window_size-1:], throughput_rewards_10_12, color="green", label=r"$10^{-12}$")
axis[0,1].plot(timesteps_10_15[window_size-1:], throughput_rewards_10_15, color="red", label=r"$10^{-15}$")
axis[0,1].plot(timesteps_10_15[window_size-1:], throughput_rewards_10_18, color="blue", label=r"$10^{-18}$")
#axis[0,1].plot(timesteps_10_15[window_size-1:], throughput_rewards_10_21, color="blue", label=r"$10^{-21}$")
#axis[0,1].plot(timesteps_10_15[window_size-1:], throughput_rewards_10_24, color="black", label=r"$10^{-24}$")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,1].set_title('Data Rate')
axis[0,1].grid()
axis[0,1].set_xlabel('Timestep')
axis[0,1].legend(loc="lower right")

#axis[1,0].plot(timesteps_10_12[window_size-1:], energy_rewards_10_12, color="green", label=r"$10^{-12}$")
axis[1,0].plot(timesteps_10_15[window_size-1:], energy_rewards_10_15, color="red", label=r"$10^{-15}$")
axis[1,0].plot(timesteps_10_15[window_size-1:], energy_rewards_10_18, color="blue", label=r"$10^{-18}$")
#axis[1,0].plot(timesteps_10_15[window_size-1:], energy_rewards_10_21, color="blue", label=r"$10^{-21}$")
#axis[1,0].plot(timesteps_10_15[window_size-1:], energy_rewards_10_24, color="black", label=r"$10^{-24}$")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[1,0].set_title('Energy Consumption')
axis[1,0].grid()
axis[1,0].set_xlabel('Timestep')
axis[1,0].legend(loc="lower right")

#axis[1,1].plot(timesteps_10_12[window_size-1:], delay_rewards_10_12, color="green", label=r"$10^{-12}$")
axis[1,1].plot(timesteps_10_15[window_size-1:], delay_rewards_10_15, color="red", label=r"$10^{-15}$")
axis[1,1].plot(timesteps_10_15[window_size-1:], delay_rewards_10_18, color="blue", label=r"$10^{-18}$")
#axis[1,1].plot(timesteps_10_15[window_size-1:], delay_rewards_10_21, color="blue", label=r"$10^{-21}$")
#axis[1,1].plot(timesteps_10_15[window_size-1:], delay_rewards_10_24, color="black", label=r"$10^{-24}$")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[1,1].set_title('Delay')
axis[1,1].grid()
axis[1,1].set_xlabel('Timestep')
axis[1,1].legend(loc="lower right")

plt.tight_layout()
plt.show()

###############################################################################################################################################################

# ###############Model Trained with 10**(-15)###############
# #energy_constant = [-24, -21, -18, -15, -12]
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier = np.log10(delay_multiplier)
# reward_values_10_15 = [89414198.400063,89021978.853357,85109006.746785,45989105.574415,-345204543.683362,-4256168917.276906]
# energy_values_10_15 = [0.001005,0.001005,0.001005,0.001005,0.001005,0.001005]
# throughput_values_10_15 = [28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193]
# delay_values_10_15 = [43.463607,43.465526,43.459549,43.472941,43.488120,43.464101]
# av_local_queue_lengths_bits_10_15 = [900.2187218721873,900.0257425742576,900.198289828983,900.1859585958595,900.0872187218721,900.2540954095409]
# av_offload_queue_lengths_bits_10_15 = [10760.869576957695,10762.193699369936,10762.475967596758,10761.68604860486,10762.38397839784,10762.86390639064]
# av_local_queue_lengths_tasks_10_15 = [3.217371737173717,3.216471647164717,3.218541854185418,3.217371737173717,3.217371737173717,3.2182718271827184]
# av_offload_queue_lengths_tasks_10_15 = [22.026012601260128,22.03015301530153,22.02997299729973,22.02907290729073,22.02925292529253,22.02961296129613]
# av_offlaoding_ratios_10_15 = [0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815]




# ###############Model Trained with 10**(-18)###############
# #energy_constant = [-24, -21, -18, -15, -12]
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier = np.log10(delay_multiplier)
# reward_values_10_18 = [84105526.442742,83347828.721909,75769349.471185,120072.525794,-756559239.790893,-8331080317.499292]
# energy_values_10_18 = [0.000602,0.000602,0.000602,0.000602,0.000602,0.000602]
# throughput_values_10_18 = [23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902]
# delay_values_10_18 = [84.220683,84.057094,84.139638,84.116089,84.114519,84.055431]
# av_local_queue_lengths_bits_10_18 = [900.1045004500451,900.2906390639063,900.05400540054,900.3936093609361,900.1348334833482,900.1957695769578]
# av_offload_queue_lengths_bits_10_18 = [24845.252745274527,24845.322862286233,24844.84320432043,24846.344464446443,24847.44779477948,24846.84194419442]
# av_local_queue_lengths_tasks_10_18 = [3.2174617461746178,3.2174617461746178,3.218001800180018,3.218721872187219,3.2171917191719177,3.2167416741674164]
# av_offload_queue_lengths_tasks_10_18 = [50.79324932493248,50.79297929792978,50.79072907290729,50.79072907290729,50.793249324932496,50.79423942394239]
# av_offlaoding_ratios_10_18 = [0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919]

# import matplotlib.pyplot as plt
# import numpy as np

# # Energy constant values (logarithmic scale)
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier_log = np.log10(delay_multiplier)

# # Data for models trained at different constants
# metrics = {
#     "Reward": [reward_values_10_15, reward_values_10_18],
#     "Energy": [energy_values_10_15, energy_values_10_18],
#     "Throughput": [throughput_values_10_15, throughput_values_10_18],
#     "Delay": [delay_values_10_15, delay_values_10_18],
#     "Local Queue Length (bits)": [av_local_queue_lengths_bits_10_15, av_local_queue_lengths_bits_10_18],
#     "Offload Queue Length (bits)": [av_offload_queue_lengths_bits_10_15, av_offload_queue_lengths_bits_10_18],
#     "Local Queue Length (tasks)": [av_local_queue_lengths_tasks_10_15, av_local_queue_lengths_tasks_10_18],
#     "Offload Queue Length (tasks)": [av_offload_queue_lengths_tasks_10_15, av_offload_queue_lengths_tasks_10_18],
#     "Offloading Ratio": [av_offlaoding_ratios_10_15, av_offlaoding_ratios_10_18],
# }

# # Model labels
# model_labels = ["Trained at 10^(-15)", "Trained at 10^(-18)"]

# # Create subplots
# fig, axs = plt.subplots(3, 3, figsize=(18, 12))
# axs = axs.flatten()

# for idx, (metric_name, metric_data) in enumerate(metrics.items()):
#     for model_idx, model_data in enumerate(metric_data):
#         axs[idx].plot(delay_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
#     axs[idx].set_title(metric_name)
#     axs[idx].set_xlabel("Log10(Delay Multiplier)")
#     axs[idx].set_ylabel(metric_name)
#     axs[idx].grid(True)
#     axs[idx].legend()

# plt.tight_layout()
# plt.show()



###############Model Trained with 10**(-15) and q_energy = 10**11###############
#q_multipliers for constraints = 0
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier = np.log10(delay_multiplier)
# reward_values_10_15 = [-71830619.826666,-72224758.808126,-76137653.153032,-115259729.470653,-506518806.412307,-4419045101.887218]
# energy_values_10_15 = [0.001005,0.001005,0.001005,0.001005,0.001005,0.001005]
# throughput_values_10_15 = [28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193]
# delay_values_10_15 = [43.463607,43.465526,43.459549,43.472941,43.488120,43.464101]
# av_local_queue_lengths_bits_10_15 = [900.2187218721873,900.0257425742576,900.198289828983,900.1859585958595,900.0872187218721,900.2540954095409]
# av_offload_queue_lengths_bits_10_15 = [10760.869576957695,10762.193699369936,10762.475967596758,10761.68604860486,10762.38397839784,10762.86390639064]
# av_local_queue_lengths_tasks_10_15 = [3.217371737173717,3.216471647164717,3.218541854185418,3.217371737173717,3.217371737173717,3.2182718271827184]
# av_offload_queue_lengths_tasks_10_15 = [22.026012601260128,22.03015301530153,22.02997299729973,22.02907290729073,22.02925292529253,22.02961296129613]
# av_offlaoding_ratios_10_15 = [0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815]




# ###############Model Trained with 10**(-18) and q_energy = 10**11###############
# #q_multipliers for constraints = 0
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier = np.log10(delay_multiplier)
# reward_values_10_18 = [-36731604.160642,-37482458.414480,-45060510.130522,-120848107.754200,-877932375.675033,-8455555383.851169]
# energy_values_10_18 = [0.000602,0.000602,0.000602,0.000602,0.000602,0.000602]
# throughput_values_10_18 = [23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902]
# delay_values_10_18 = [84.220683,84.057094,84.139638,84.116089,84.114519,84.055431]
# av_local_queue_lengths_bits_10_18 = [900.1045004500451,900.2906390639063,900.05400540054,900.3936093609361,900.1348334833482,900.1957695769578]
# av_offload_queue_lengths_bits_10_18 = [24845.252745274527,24845.322862286233,24844.84320432043,24846.344464446443,24847.44779477948,24846.84194419442]
# av_local_queue_lengths_tasks_10_18 = [3.2174617461746178,3.2174617461746178,3.218001800180018,3.218721872187219,3.2171917191719177,3.2167416741674164]
# av_offload_queue_lengths_tasks_10_18 = [50.79324932493248,50.79297929792978,50.79072907290729,50.79072907290729,50.793249324932496,50.79423942394239]
# av_offlaoding_ratios_10_18 = [0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919]

# import matplotlib.pyplot as plt
# import numpy as np

# # Energy constant values (logarithmic scale)
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier_log = np.log10(delay_multiplier)

# # Data for models trained at different constants
# metrics = {
#     "Reward": [reward_values_10_15, reward_values_10_18],
#     "Energy": [energy_values_10_15, energy_values_10_18],
#     "Throughput": [throughput_values_10_15, throughput_values_10_18],
#     "Delay": [delay_values_10_15, delay_values_10_18],
#     "Local Queue Length (bits)": [av_local_queue_lengths_bits_10_15, av_local_queue_lengths_bits_10_18],
#     "Offload Queue Length (bits)": [av_offload_queue_lengths_bits_10_15, av_offload_queue_lengths_bits_10_18],
#     "Local Queue Length (tasks)": [av_local_queue_lengths_tasks_10_15, av_local_queue_lengths_tasks_10_18],
#     "Offload Queue Length (tasks)": [av_offload_queue_lengths_tasks_10_15, av_offload_queue_lengths_tasks_10_18],
#     "Offloading Ratio": [av_offlaoding_ratios_10_15, av_offlaoding_ratios_10_18],
# }

# # Model labels
# model_labels = ["Trained at 10^(-15)", "Trained at 10^(-18)"]

# # Create subplots
# fig, axs = plt.subplots(3, 3, figsize=(18, 12))
# axs = axs.flatten()

# for idx, (metric_name, metric_data) in enumerate(metrics.items()):
#     for model_idx, model_data in enumerate(metric_data):
#         axs[idx].plot(delay_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
#     axs[idx].set_title(metric_name)
#     axs[idx].set_xlabel("Log10(Delay Multiplier)")
#     axs[idx].set_ylabel(metric_name)
#     axs[idx].grid(True)
#     axs[idx].legend()

# plt.tight_layout()
# plt.show()

###############Model Trained with 10**(-18) and q_energy = 10**11###############
#q_multipliers for constraints = 0
# Model trained with q_delay = 10**5 and q_energy = 1.5*10**10
# delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
# delay_multiplier = np.log10(delay_multiplier)
# reward_values_10_18 = [-62224814.375452,-62659832.250039,-67009729.895954,-110524757.685361,-545585704.729342,-4896568725.535871]
# energy_values_10_18 = [0.000917,0.000917,0.000917,0.000917,0.000917,0.000917]
# throughput_values_10_18 = [29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391]
# delay_values_10_18 = [48.345832,48.345832,48.345832,48.345832,48.345832,48.345832]
# av_local_queue_lengths_bits_10_18 = [162.1866786678668,162.1866786678668,162.1866786678668,162.1866786678668,162.1866786678668,162.1866786678668]
# av_offload_queue_lengths_bits_10_18 = [15185.75670567057,15185.75670567057,15185.75670567057,15185.75670567057,15185.75670567057,15185.75670567057]
# av_local_queue_lengths_tasks_10_18 = [0.7893789378937893,0.7893789378937893,0.7893789378937893,0.7893789378937893,0.7893789378937893,0.7893789378937893]
# av_offload_queue_lengths_tasks_10_18 = [29.022952295229526,29.022952295229526,29.022952295229526,29.022952295229526,29.022952295229526,29.022952295229526]
# av_offlaoding_ratios_10_18 = [0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
delay_multiplier = np.log10(delay_multiplier)
reward_values_10_18 = [-60845682.911375,-62868679.580698,-65098334.522792,-102779143.407505,-410190658.411065,-3390124134.804518]
energy_values_10_18 = [0.000916,0.000914,0.000922,0.000902,0.000905,0.000922]
throughput_values_10_18 = [30844227.444618,29059782.487484,31660695.877396,31080246.089361,32240086.330267,32500901.939190]
delay_values_10_18 = [42.919048,50.831566,45.513450,43.660912,35.190814,33.304342]
av_local_queue_lengths_bits_10_18 = [161.75328532853288,158.84347434743475,161.88739873987402,153.24761476147614,168.55121512151214,162.22115211521157]
av_offload_queue_lengths_bits_10_18 = [13117.235463546354,15791.38424842484,13973.904500450048,13107.302610261024,10176.369936993699,9657.831323132314]
av_local_queue_lengths_tasks_10_18 = [0.8015301530153015,0.7619261926192619,0.7696669666966698,0.7519351935193519,0.8286228622862286,0.814941494149415]
av_offload_queue_lengths_tasks_10_18 = [25.112691269126913,30.327092709270918,26.78307830783078,25.147524752475245,19.616291629162916,18.5991899189919]
av_offlaoding_ratios_10_18 = [0.8808400909333927,0.8761801161460895,0.8798378369183844,0.8800201242734533,0.8800201242734533,0.8802840339597016]
import matplotlib.pyplot as plt
import numpy as np

# Energy constant values (logarithmic scale)
delay_multiplier = [10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8)]
delay_multiplier_log = np.log10(delay_multiplier)

# Data for models trained at different constants
metrics = {
    "Reward": [reward_values_10_18],
    "Energy": [energy_values_10_18],
    "Throughput": [throughput_values_10_18],
    "Delay": [delay_values_10_18],
    "Local Queue Length (bits)": [av_local_queue_lengths_bits_10_18],
    "Offload Queue Length (bits)": [av_offload_queue_lengths_bits_10_18],
    "Local Queue Length (tasks)": [av_local_queue_lengths_tasks_10_18],
    "Offload Queue Length (tasks)": [av_offload_queue_lengths_tasks_10_18],
    "Offloading Ratio": [av_offlaoding_ratios_10_18],
}

# Model labels
model_labels = ["Trained at 10^(-18)"]

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
axs = axs.flatten()

for idx, (metric_name, metric_data) in enumerate(metrics.items()):
    for model_idx, model_data in enumerate(metric_data):
        axs[idx].plot(delay_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
    axs[idx].set_title(metric_name)
    axs[idx].set_xlabel("Log10(Delay Multiplier)")
    axs[idx].set_ylabel(metric_name)
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()
