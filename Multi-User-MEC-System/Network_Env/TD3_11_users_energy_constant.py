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

###############Model Trained with 10**(-15)###############
#energy_constant = [-24, -21, -18, -15, -12]
energy_constant = [10**(-24), 10**(-21), 10**(-18), 10**(-15), 10**(-12)]
energy_constant = np.log10(energy_constant)
reward_values_10_15 = [8924899196.160061,8925004431.572468,8922624822.838854,6358059133.662412,-2557953459713.360352]
energy_values_10_15 = [0.000656,0.000656,0.000910,0.254810,254.147045]
throughput_values_10_15 = [29545422.636385,29545422.636385,29545422.636385,29545422.636385,29545422.636385]
delay_values_10_15 = [46.601752,46.593284,46.572479,46.597560,46.571995]
av_local_queue_lengths_bits_10_15 = [154.91557155715572,154.8925292529253,154.8815481548155,154.88937893789378,154.86714671467146]
av_offload_queue_lengths_bits_10_15 = [14080.715751575159,14080.52502250225,14079.038523852387,14079.269936993696,14078.868226822682]
av_local_queue_lengths_tasks_10_15 = [0.7516651665166517,0.7519351935193519,0.7518451845184518,0.752025202520252,0.751845184518452]
av_offload_queue_lengths_tasks_10_15 = [27.11260126012601,27.11206120612061,27.10738073807381,27.10792079207921,27.10828082808281]
av_offlaoding_ratios_10_15 = [0.8762106628304448,0.8762106628304448,0.8762106628304448,0.8762106628304448,0.8762106628304448]




###############Model Trained with 10**(-18)###############
#energy_constant = [-24, -21, -18, -15, -12]
energy_constant = [10**(-24), 10**(-21), 10**(-18), 10**(-15), 10**(-12)]
energy_constant = np.log10(energy_constant)
reward_values_10_18 = [7679569282.281820,7680390457.318504,7678775398.632202,5113748730.147283,-2559160754715.782227]
energy_values_10_18 = [0.000161,0.000161,0.000415,0.254310,254.142905]
throughput_values_10_18 = [23012359.095385,23012359.095385,23012359.095385,23012359.095385,23012359.095385]
delay_values_10_18 = [104.004887,103.925276,103.830318,103.897612,103.878326]
av_local_queue_lengths_bits_10_18 = [154.8585058505851,154.85445544554457,154.8861386138614,154.86228622862285,154.85364536453648]
av_offload_queue_lengths_bits_10_18 = [34624.724752475246,34627.77632763276,34627.0300630063,34627.340774077406,34626.551125112506]
av_local_queue_lengths_tasks_10_18 = [0.7516651665166517,0.7516651665166516,0.7514851485148515,0.7515751575157517,0.7515751575157517]
av_offload_queue_lengths_tasks_10_18 = [66.24671467146716,66.24995499549955,66.24788478847884,66.24833483348336,66.24950495049505]
av_offlaoding_ratios_10_18 = [0.8800333131011141,0.8800333131011141,0.8800333131011141,0.8800333131011141,0.8800333131011141]

import matplotlib.pyplot as plt
import numpy as np

# Energy constant values (logarithmic scale)
energy_constant = [10**(-24), 10**(-21), 10**(-18), 10**(-15), 10**(-12)]
energy_constant_log = np.log10(energy_constant)

# Data for models trained at different constants
metrics = {
    "Reward": [reward_values_10_15, reward_values_10_18],
    "Energy": [energy_values_10_15, energy_values_10_18],
    "Throughput": [throughput_values_10_15, throughput_values_10_18],
    "Delay": [delay_values_10_15, delay_values_10_18],
    "Local Queue Length (bits)": [av_local_queue_lengths_bits_10_15, av_local_queue_lengths_bits_10_18],
    "Offload Queue Length (bits)": [av_offload_queue_lengths_bits_10_15, av_offload_queue_lengths_bits_10_18],
    "Local Queue Length (tasks)": [av_local_queue_lengths_tasks_10_15, av_local_queue_lengths_tasks_10_18],
    "Offload Queue Length (tasks)": [av_offload_queue_lengths_tasks_10_15, av_offload_queue_lengths_tasks_10_18],
    "Offloading Ratio": [av_offlaoding_ratios_10_15, av_offlaoding_ratios_10_18],
}

# Model labels
model_labels = ["Trained at 10^(-15)", "Trained at 10^(-18)"]

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
axs = axs.flatten()

for idx, (metric_name, metric_data) in enumerate(metrics.items()):
    for model_idx, model_data in enumerate(metric_data):
        axs[idx].plot(energy_constant_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
    axs[idx].set_title(metric_name)
    axs[idx].set_xlabel("Log10(Energy Constant)")
    axs[idx].set_ylabel(metric_name)
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()
