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

# ###############################################################################################################################################################

# ###############Model Trained with 10**(-15)###############
# #energy_constant = [-24, -21, -18, -15, -12]
# energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
# energy_multiplier = np.log10(energy_multiplier)
# reward_values_10_15 = [85212764.450489,85212569.562105,85202401.857173,85112064.304178,84204368.022686,75157143.653230,-15316049.115504,-920075838.000343,-9967344857.867300]
# energy_values_10_15 = [0.001005,0.001005,0.001005,0.001005,0.001005,0.001005,0.001005,0.001005,0.001005]
# throughput_values_10_15 = [28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193]
# delay_values_10_15 = [43.462750,43.458445,43.469351,43.471706,43.474202,43.459148,43.452644,43.467450,43.456249]
# av_local_queue_lengths_bits_10_15 = [900.1996399639963,900.0831683168316,900.4216021602159,900.1374437443744,900.1677767776778,900.0567056705672,900.061206120612,900.1199819981998,900.3808280828084]
# av_offload_queue_lengths_bits_10_15 = [10762.139963996398,10762.694419441945,10762.28883888389,10763.430513051304,10763.472367236724,10762.796129612962,10763.020792079209,10762.319351935193,10761.39090909091]
# av_local_queue_lengths_tasks_10_15 = [3.218541854185419,3.2184518451845183,3.218361836183618,3.218541854185418,3.216921692169217,3.2169216921692168,3.2166516651665167,3.217551755175518,3.218991899189919]
# av_offload_queue_lengths_tasks_10_15 = [22.02781278127813,22.03024302430243,22.02763276327633,22.03240324032403,22.031233123312333,22.02907290729073,22.02898289828983,22.027542754275427,22.02682268226823]
# av_offlaoding_ratios_10_15 = [0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815]




# ###############Model Trained with 10**(-18)###############
# #energy_constant = [-24, -21, -18, -15, -12]
# energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
# energy_multiplier = np.log10(energy_multiplier)
# reward_values_10_18 = [75838536.961637,75834713.777456,75838962.103145,75768536.115302,75237111.325976,69826091.206807,15678697.306737,-525883761.641128,-5940101589.447633]
# energy_values_10_18 = [0.000601,0.000602,0.000602,0.000602,0.000602,0.000602,0.000602,0.000602,0.000602]
# throughput_values_10_18 = [23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902]
# delay_values_10_18 = [84.161987,84.141673,84.154080,84.088846,84.088846,84.154237,84.159683,84.143086,84.100417]
# av_local_queue_lengths_bits_10_18 = [900.2670567056706,900.1442844284429,900.1608460846085,900.105400540054,900.1938793879389,900.3060306030602,900.0092709270928,900.1194419441944,900.1944194419441]
# av_offload_queue_lengths_bits_10_18 = [24847.862556255623,24845.181728172818,24845.744464446445,24845.241944194422,24845.35454545455,24847.043024302428,24844.963366336633,24847.35544554456,24846.189378937896]
# av_local_queue_lengths_tasks_10_18 = [3.218001800180018,3.2181818181818187,3.2167416741674164,3.2174617461746178,3.217551755175518,3.2171917191719173,3.2172817281728174,3.218361836183618,3.2174617461746178]
# av_offload_queue_lengths_tasks_10_18 = [50.793429342934296,50.79036903690369,50.79297929792979,50.78883888388839,50.7914491449145,50.79297929792979,50.789918991899185,50.7947794779478,50.7930693069307]
# av_offlaoding_ratios_10_18 = [0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919]

# import matplotlib.pyplot as plt
# import numpy as np

# # Energy constant values (logarithmic scale)
# energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
# energy_multiplier_log = np.log10(energy_multiplier)

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
#         axs[idx].plot(energy_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
#     axs[idx].set_title(metric_name)
#     axs[idx].set_xlabel("Log10(Energy Multiplier)")
#     axs[idx].set_ylabel(metric_name)
#     axs[idx].grid(True)
#     axs[idx].legend()

# plt.tight_layout()
# plt.show()


###############################################################################################################################################################

###############Model Trained with 10**(-15)###############
###############q_delay = 10**6, contraint q_multipliers = 0############################
#energy_constant = [-24, -21, -18, -15, -12]
# energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
# energy_multiplier = np.log10(energy_multiplier)
# reward_values_10_15 = [-14734712.563343,-14728636.668085,-14728314.639455,-14825335.733828,-15734010.033541,-24770785.754939,-115254137.341413,-1020018484.513201,-10067502015.017153]
# energy_values_10_15 = [0.001005,0.001005,0.001005,0.001005,0.001005,0.001005,0.001005,0.001005,0.001005]
# throughput_values_10_15 = [28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193,28740962.559193]
# delay_values_10_15 = [43.475575,43.475575,43.475575,43.475575,43.475575,43.475575,43.475575,43.475575,43.475575]
# av_local_queue_lengths_bits_10_15 = [900.2891089108912,900.2891089108912,900.2891089108912,900.2891089108912,900.2891089108912,900.2891089108912,900.2891089108912,900.2891089108912,900.2891089108912]
# av_offload_queue_lengths_bits_10_15 = [10762.824392439243,10762.824392439243,10762.824392439243,10762.824392439243,10762.824392439243,10762.824392439243,10762.824392439243,10762.824392439243,10762.824392439243]
# av_local_queue_lengths_tasks_10_15 = [3.217821782178218,3.217821782178218,3.217821782178218,3.217821782178218,3.217821782178218,3.217821782178218,3.217821782178218,3.217821782178218,3.217821782178218]
# av_offload_queue_lengths_tasks_10_15 = [22.029432943294328,22.029432943294328,22.029432943294328,22.029432943294328,22.029432943294328,22.029432943294328,22.029432943294328,22.029432943294328,22.029432943294328]
# av_offlaoding_ratios_10_15 = [0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815,0.8008939749509815]




###############Model Trained with 10**(-18)###############
###############q_delay = 10**6, contraint q_multipliers = 0############################
# energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
# energy_multiplier = np.log10(energy_multiplier)
# reward_values_10_18 = [-60605618.506280,-60648619.599805,-60656538.796664,-60678562.241029,-61223740.132114,-66677881.727360,-120775016.104925,-662142735.958032,-6076520231.419865]
# energy_values_10_18 = [0.000602,0.000602,0.000602,0.000602,0.000602,0.000602,0.000602,0.000602,0.000602]
# throughput_values_10_18 = [23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902,23522071.229902]
# delay_values_10_18 = [84.127630,84.127630,84.127630,84.127630,84.127630,84.127630,84.127630,84.127630,84.127630]
# av_local_queue_lengths_bits_10_18 = [900.3362736273629,900.3362736273629,900.3362736273629,900.3362736273629,900.3362736273629,900.3362736273629,900.3362736273629,900.3362736273629,900.3362736273629]
# av_offload_queue_lengths_bits_10_18 = [24845.5097209721,24845.5097209721,24845.5097209721,24845.5097209721,24845.5097209721,24845.5097209721,24845.5097209721,24845.5097209721,24845.5097209721]
# av_local_queue_lengths_tasks_10_18 = [3.218001800180018,3.218001800180018,3.218001800180018,3.218001800180018,3.218001800180018,3.218001800180018,3.218001800180018,3.218001800180018,3.218001800180018]
# av_offload_queue_lengths_tasks_10_18 = [50.793519351935196,50.793519351935196,50.793519351935196,50.793519351935196,50.793519351935196,50.793519351935196,50.793519351935196,50.793519351935196,50.793519351935196]
# av_offlaoding_ratios_10_18 = [0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919,0.8039697960805919]

# import matplotlib.pyplot as plt
# import numpy as np

# # Energy constant values (logarithmic scale)
# energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
# energy_multiplier_log = np.log10(energy_multiplier)

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
#         axs[idx].plot(energy_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
#     axs[idx].set_title(metric_name)
#     axs[idx].set_xlabel("Log10(Energy Multiplier)")
#     axs[idx].set_ylabel(metric_name)
#     axs[idx].grid(True)
#     axs[idx].legend()

# plt.tight_layout()
# plt.show()

##############Model Trained with 10**(-18)###############
##############q_delay = 10**6, contraint q_multipliers = 0############################
# Model trained with q_delay = 10**5 and q_energy = 1.5*10**10
energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
energy_multiplier = np.log10(energy_multiplier)
reward_values_10_18 = [-17041107.217397,-18829336.664588,-18824065.219941,-18906526.911992,-19725869.718402,-27984240.010765,-110535948.774576,-935867112.472442,-9189124028.289331]
energy_values_10_18 = [0.000910,0.000917,0.000917,0.000917,0.000917,0.000917,0.000917,0.000917,0.000917]
throughput_values_10_18 = [29545422.636385,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391]
delay_values_10_18 = [46.586439,48.355495,48.341970,48.341899,48.341899,48.341899,48.341899,48.341899]
av_local_queue_lengths_bits_10_18 = [154.8940594059406,162.14761476147615,162.1854185418542,162.14752475247522,162.2141314131413,162.2141314131413,162.2141314131413,162.2141314131413,162.2141314131413]
av_offload_queue_lengths_bits_10_18 = [14080.10585058506,15185.49909990999,15185.066156615661,15185.083798379836,15184.421422142213,15184.421422142213,15184.421422142213,15184.421422142213,15184.421422142213]
av_local_queue_lengths_tasks_10_18 = [0.7521152115211521,0.7889288928892889,0.7892889288928893,0.7891089108910893,0.7896489648964896,0.7896489648964896,0.7896489648964896,0.7896489648964896,0.7896489648964896]
av_offload_queue_lengths_tasks_10_18 = [27.11242124212421,29.02079207920792,29.021332133213324,29.021602160216016,29.017911791179117,29.017911791179117,29.017911791179117,29.017911791179117,29.017911791179117]
av_offlaoding_ratios_10_18 = [0.8794931351275455,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

reward_values_10_18 = [9051919.073558,5399579.124751,2019451.288287,485791.742718,-12254103.294576,1611751.063717,-98757939.652075,-914327172.240875,-9238936478.026430]
energy_values_10_18 = [0.000920,0.000927,0.000910,0.000916,0.000922,0.000908,0.000917,0.000916,0.000923]
throughput_values_10_18 = [35816911.535342,34280753.608881,33650421.782139,33945544.282084,33201117.540544,35851695.331865,31405140.858318,33464747.883064,31897005.926268]
delay_values_10_18 = [26.764900,28.880248,31.621875,33.368165,44.532986,25.159490,38.423520,32.260157,41.864523]
av_local_queue_lengths_bits_10_18 = [160.07173717371737,148.383798379838,167.01332133213322,169.1065706570657,194.1058505850585,162.5067506750675,161.66462646264628,157.9016201620162,162.3468946894689]
av_offload_queue_lengths_bits_10_18 = [7045.460126012602,7996.253825382539,8930.684068406841,9476.43402340234,13381.089738973898,6501.814941494149,11366.32205220522,9267.501530153017,12981.893159315932]
av_local_queue_lengths_tasks_10_18 = [0.7692169216921694,0.7282628262826283,0.8137713771377139,0.8059405940594059,0.9408640864086408,0.8015301530153015,0.7892889288928895,0.7648064806480648,0.801890189018902]
av_offload_queue_lengths_tasks_10_18 = [13.700810081008102,15.431323132313231,17.252475247524753,18.309810981098106,25.71008100810081,12.643384338433844,21.933303330333032,17.80891089108911,24.816381638163815]
av_offlaoding_ratios_10_18 = [0.8769606779371077,0.8788582792689191,0.8797917740525499,0.8796764302391434,0.8762767385828156,0.8788208960423418,0.8776563347058645,0.8789123117555402,0.8802207489221867]

import matplotlib.pyplot as plt
import numpy as np
# Energy constant values (logarithmic scale)
energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
energy_multiplier_log = np.log10(energy_multiplier)

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
        axs[idx].plot(energy_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
    axs[idx].set_title(metric_name)
    axs[idx].set_xlabel("Log10(Energy Multiplier)")
    axs[idx].set_ylabel(metric_name)
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()

##############Model Trained with 10**(-18)###############
##############q_delay = 10**6, contraint q_multipliers = 0############################
# Model trained with q_delay = 10**5 and q_energy = 1.5*10**10
energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
energy_multiplier = np.log10(energy_multiplier)
reward_values_10_18 = [-17041107.217397,-18829336.664588,-18824065.219941,-18906526.911992,-19725869.718402,-27984240.010765,-110535948.774576,-935867112.472442,-9189124028.289331]
energy_values_10_18 = [0.000910,0.000917,0.000917,0.000917,0.000917,0.000917,0.000917,0.000917,0.000917]
throughput_values_10_18 = [29545422.636385,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391,29527075.334391]
delay_values_10_18 = [46.586439,48.355495,48.341970,48.341899,48.341899,48.341899,48.341899,48.341899]
av_local_queue_lengths_bits_10_18 = [154.8940594059406,162.14761476147615,162.1854185418542,162.14752475247522,162.2141314131413,162.2141314131413,162.2141314131413,162.2141314131413,162.2141314131413]
av_offload_queue_lengths_bits_10_18 = [14080.10585058506,15185.49909990999,15185.066156615661,15185.083798379836,15184.421422142213,15184.421422142213,15184.421422142213,15184.421422142213,15184.421422142213]
av_local_queue_lengths_tasks_10_18 = [0.7521152115211521,0.7889288928892889,0.7892889288928893,0.7891089108910893,0.7896489648964896,0.7896489648964896,0.7896489648964896,0.7896489648964896,0.7896489648964896]
av_offload_queue_lengths_tasks_10_18 = [27.11242124212421,29.02079207920792,29.021332133213324,29.021602160216016,29.017911791179117,29.017911791179117,29.017911791179117,29.017911791179117,29.017911791179117]
av_offlaoding_ratios_10_18 = [0.8794931351275455,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897,0.8800782795195897]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

reward_values_10_18 = []
energy_values_10_18 = []
throughput_values_10_18 = []
delay_values_10_18 = []
av_offload_queue_lengths_bits_10_18 = []
av_local_queue_lengths_tasks_10_18 = []
av_offload_queue_lengths_tasks_10_18 = []
av_offlaoding_ratios_10_18 = []

import matplotlib.pyplot as plt
import numpy as np
# Energy constant values (logarithmic scale)
energy_multiplier = [10**(5), 10**(6), 10**(7), 10**(8), 10**(9), 10**(10),10**(11),10**(12),10**(13)]
energy_multiplier_log = np.log10(energy_multiplier)

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
        axs[idx].plot(energy_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
    axs[idx].set_title(metric_name)
    axs[idx].set_xlabel("Log10(Energy Multiplier)")
    axs[idx].set_ylabel(metric_name)
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()
