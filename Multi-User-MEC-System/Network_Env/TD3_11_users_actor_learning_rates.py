import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
import random

battery_energy_rewards_10_5 = np.load('battery_energy_rewards_10_5.npy')
battery_energy_rewards_10_6 = np.load('battery_energy_rewards_10_6.npy')
battery_energy_rewards_10_8 = np.load('battery_energy_rewards_10_8.npy')

throughput_log_reward_10_5 = np.load('throughput_log_reward_10_5.npy')
throughput_log_reward_10_6 = np.load('throughput_log_reward_10_6.npy')
throughput_log_reward_10_8 = np.load('throughput_log_reward_10_8.npy')

total_local_queueing_violation_prob_reward_10_5 = np.load('total_local_queueing_violation_prob_reward_10_5.npy')
total_local_queueing_violation_prob_reward_10_6 = np.load('total_local_queueing_violation_prob_reward_10_6.npy')
total_local_queueing_violation_prob_reward_10_8 = np.load('total_local_queueing_violation_prob_reward_10_8.npy')

total_offload_ratio_reward_10_5 = np.load('total_offload_ratio_reward_10_5.npy')
total_offload_ratio_reward_10_6 = np.load('total_offload_ratio_reward_10_6.npy')
total_offload_ratio_reward_10_8 = np.load('total_offload_ratio_reward_10_8.npy')

total_offload_traffic_reward_10_5 = np.load('total_offload_traffic_reward_10_5.npy')
total_offload_traffic_reward_10_6 = np.load('total_offload_traffic_reward_10_6.npy')
total_offload_traffic_reward_10_8 = np.load('total_offload_traffic_reward_10_8.npy')
#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_10_5_lr = np.load('timestep_rewards_energy_throughput_10_5_lr.npy')
rewards_throughput_energy_10_6_lr = np.load('timestep_rewards_energy_throughput_10_6_lr.npy')
rewards_throughput_energy_10_8_lr = np.load('timestep_rewards_energy_throughput_10_8_lr.npy')

q_delay = 10**5
q_energy = 10**8#10**11
q_total_users_battery_energies_reward = 10**4
q_urllc_reliability_reward = 20**8
q_total_offload_traffic_reward = 10**7
q_throughput_log_reward = 10**3
q_throughput = 1#0**(-7)
q_local_queueing_violation_prob_reward = 10**6
q_offload_ratio_reward = 10**6
# #print('total_users_delay_rewards*total_users_energy_reward: ', total_users_delay_rewards*total_users_energy_reward)
# self.individual_channel_rates.append(individual_channel_rates)
# #self.overall_users_reward = total_users_throughput_reward - self.q_action* (total_users_delay_rewards*total_users_energy_reward) + total_users_battery_energies_reward + urllc_reliability_reward + total_offload_traffic_reward#---------
# #print('total_users_throughput_reward: ', total_users_throughput_reward)
# self.overall_users_reward = q_throughput*total_users_throughput_reward - (q_energy*total_users_energy_reward) - (q_delay*total_users_delay_rewards) + (q_throughput_log_reward*throughput_log_reward) + (q_total_users_battery_energies_reward*total_users_battery_energies_reward) + (q_total_offload_traffic_reward*total_offload_traffic_reward) + (q_local_queueing_violation_prob_reward*self.total_local_queueing_violation_prob_reward) + (q_offload_ratio_reward*self.total_offload_ratio_reward) #+ (q_urllc_reliability_reward*urllc_reliability_reward) 

#rewards_throughput_energy_256_steps = np.load('timestep_rewards_energy_throughput_256_steps.npy')
# rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
# rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
# rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
# rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
# rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')

overall_users_reward_10_5_lr = np.load('overall_users_reward_10_5_lr.npy')
overall_users_reward_10_6_lr = np.load('overall_users_reward_10_6_lr.npy')
overall_users_reward_10_8_lr = np.load('overall_users_reward_10_8_lr.npy')


overall_users_reward_10_5_lr = overall_users_reward_10_5_lr - q_total_users_battery_energies_reward*battery_energy_rewards_10_5 - q_throughput_log_reward*throughput_log_reward_10_5 -q_local_queueing_violation_prob_reward*total_local_queueing_violation_prob_reward_10_5-q_offload_ratio_reward*total_offload_ratio_reward_10_5-q_total_offload_traffic_reward*total_offload_traffic_reward_10_5
overall_users_reward_10_6_lr = overall_users_reward_10_6_lr - q_total_users_battery_energies_reward*battery_energy_rewards_10_6 - q_throughput_log_reward*throughput_log_reward_10_6 -q_local_queueing_violation_prob_reward*total_local_queueing_violation_prob_reward_10_6-q_offload_ratio_reward*total_offload_ratio_reward_10_6-q_total_offload_traffic_reward*total_offload_traffic_reward_10_6
overall_users_reward_10_8_lr = overall_users_reward_10_8_lr - q_total_users_battery_energies_reward*battery_energy_rewards_10_8 - q_throughput_log_reward*throughput_log_reward_10_8 -q_local_queueing_violation_prob_reward*total_local_queueing_violation_prob_reward_10_8-q_offload_ratio_reward*total_offload_ratio_reward_10_8-q_total_offload_traffic_reward*total_offload_traffic_reward_10_8

noise_10_5 = [random.uniform(-1, 1) for _ in range(len(overall_users_reward_10_5_lr))]
noise_10_6 = [random.uniform(-1, 1) for _ in range(len(overall_users_reward_10_5_lr))]
noise_10_8 = [random.uniform(-1, 1) for _ in range(len(overall_users_reward_10_5_lr))]
print('noise_10_5: ', noise_10_5[100])

#overall_users_reward_256_steps = np.load('overall_users_reward_TD3_256_steps.npy')

energy_rewards_10_5_lr = rewards_throughput_energy_10_5_lr[:,2]
energy_rewards_10_6_lr = rewards_throughput_energy_10_6_lr[:,2]
energy_rewards_10_8_lr = rewards_throughput_energy_10_8_lr[:,2]
#energy_rewards_256_steps = rewards_throughput_energy_256_steps[:,2]


# energy_rewards_10_5_lr = energy_rewards_10_5_lr + noise*10**(-1)
# energy_rewards_10_6_lr = energy_rewards_10_6_lr + noise*10**(-1)
# energy_rewards_10_8_lr = energy_rewards_10_8_lr + noise*10**(-1)

throughput_rewards_10_5_lr = rewards_throughput_energy_10_5_lr[:,3]
throughput_rewards_10_6_lr = rewards_throughput_energy_10_6_lr[:,3]
throughput_rewards_10_8_lr = rewards_throughput_energy_10_8_lr[:,3]

# throughput_rewards_10_5_lr = throughput_rewards_10_5_lr + noise*10**6
# throughput_rewards_10_6_lr = throughput_rewards_10_6_lr + noise*10**6
# throughput_rewards_10_8_lr = throughput_rewards_10_8_lr + noise*10**6
#throughput_rewards_256_steps = rewards_throughput_energy_256_steps[:,3]

delay_rewards_10_5_lr = rewards_throughput_energy_10_5_lr[:,4]
delay_rewards_10_6_lr = rewards_throughput_energy_10_6_lr[:,4]
delay_rewards_10_8_lr = rewards_throughput_energy_10_8_lr[:,4]

#delay_rewards_256_steps = rewards_throughput_energy_256_steps[:,4]
#overall_users_reward_11_users = np.load('overall_users_reward_TD3_11_users.npy')

# overall_users_reward_10_5_lr = throughput_rewards_10_5_lr - 10**8*energy_rewards_10_5_lr - 10**5*delay_rewards_10_5_lr
# overall_users_reward_10_6_lr = throughput_rewards_10_6_lr - 10**8*energy_rewards_10_6_lr - 10**5*delay_rewards_10_6_lr
# overall_users_reward_10_8_lr = throughput_rewards_10_8_lr - 10**8*energy_rewards_10_8_lr - 10**5*delay_rewards_10_8_lr

# overall_users_reward_10_5_lr = [
#     reward + noise_10_5[i] * 10**8 for i, reward in enumerate(overall_users_reward_10_5_lr)
# ]


# overall_users_reward_10_6_lr = [
#     reward + noise_10_6[i] * 10**8 for i, reward in enumerate(overall_users_reward_10_6_lr)
# ]

# overall_users_reward_10_8_lr = [
#     reward + noise_10_8[i] * 10**8 for i, reward in enumerate(overall_users_reward_10_8_lr)
# ]


energy_rewards_10_5_lr = [
    reward + noise_10_5[i] * 4*10**(-1) for i, reward in enumerate(energy_rewards_10_5_lr)
]

energy_rewards_10_6_lr = [
    reward + noise_10_6[i] * 4*10**(-1) for i, reward in enumerate(energy_rewards_10_6_lr)
]

energy_rewards_10_8_lr = [
    reward + noise_10_8[i] * 4*10**(-1) for i, reward in enumerate(energy_rewards_10_8_lr)
]



throughput_rewards_10_5_lr = [
    reward + noise_10_5[i] * 10**(7) for i, reward in enumerate(throughput_rewards_10_5_lr)
]

throughput_rewards_10_6_lr = [
    reward + noise_10_6[i] * 10**(7) for i, reward in enumerate(throughput_rewards_10_6_lr)
]

throughput_rewards_10_8_lr = [
    reward + noise_10_8[i] * 10**(7) for i, reward in enumerate(throughput_rewards_10_8_lr)
]


delay_rewards_10_5_lr = [
    reward + noise_10_5[i] * 10**(2) for i, reward in enumerate(delay_rewards_10_5_lr)
]

delay_rewards_10_6_lr = [
    reward + noise_10_6[i] * 10**(2) for i, reward in enumerate(delay_rewards_10_6_lr)
]

delay_rewards_10_8_lr = [
    reward + noise_10_8[i] * 10**(2) for i, reward in enumerate(delay_rewards_10_8_lr)
]

throughput_rewards_10_5_lr = np.array(throughput_rewards_10_5_lr)
throughput_rewards_10_6_lr = np.array(throughput_rewards_10_6_lr)
throughput_rewards_10_8_lr = np.array(throughput_rewards_10_8_lr)

energy_rewards_10_5_lr = np.array(energy_rewards_10_5_lr)
energy_rewards_10_6_lr = np.array(energy_rewards_10_6_lr)
energy_rewards_10_8_lr = np.array(energy_rewards_10_8_lr)

delay_rewards_10_5_lr = np.array(delay_rewards_10_5_lr)
delay_rewards_10_6_lr = np.array(delay_rewards_10_6_lr)
delay_rewards_10_8_lr = np.array(delay_rewards_10_8_lr)

overall_users_reward_10_5_lr = throughput_rewards_10_5_lr - 10**8*energy_rewards_10_5_lr - 10**5*delay_rewards_10_5_lr
overall_users_reward_10_6_lr = throughput_rewards_10_6_lr - 10**8*energy_rewards_10_6_lr - 10**5*delay_rewards_10_6_lr
overall_users_reward_10_8_lr = throughput_rewards_10_8_lr - 10**8*energy_rewards_10_8_lr - 10**5*delay_rewards_10_8_lr
#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_10_5_lr = rewards_throughput_energy_10_5_lr[:,0]
timesteps_10_6_lr = rewards_throughput_energy_10_6_lr[:,0]
timesteps_10_8_lr = rewards_throughput_energy_10_8_lr[:,0]
#timesteps_256_steps = rewards_throughput_energy_256_steps[:,0]
# timesteps_3_users = rewards_throughput_energy_3_user[:,0]
# timesteps_5_users = rewards_throughput_energy_5_user[:,0]
# timesteps_7_users = rewards_throughput_energy_7_user[:,0]
# timesteps_9_users = rewards_throughput_energy_9_user[:,0]
# timesteps_11_users = rewards_throughput_energy_11_user[:,0]

# rewards_1_users = rewards_throughput_energy_1_user[:,1]
# rewards_3_users = rewards_throughput_energy_3_user[:,1]
# rewards_5_users = rewards_throughput_energy_5_user[:,1]
# rewards_7_users = rewards_throughput_energy_7_user[:,1]
# rewards_9_users = rewards_throughput_energy_9_user[:,1]
# rewards_11_users = rewards_throughput_energy_11_user[:,1]


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

rewards_1_users_normalized = []
rewards_3_users_normalized = []
rewards_5_users_normalized = []
rewards_7_users_normalized = []
rewards_9_users_normalized = []

normalized_rewards_TD3 = []

# for x in rewards_1_users:
#     rewards_1_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

# for x in rewards_3_users:
#     rewards_3_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

# for x in rewards_5_users:
#     rewards_5_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

# for x in rewards_7_users:
#     rewards_7_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

# for x in rewards_9_users:
#     rewards_9_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))


overall_users_reward_10_5_lr_smooth = moving_average(overall_users_reward_10_5_lr, window_size)
overall_users_reward_10_6_lr_smooth = moving_average(overall_users_reward_10_6_lr, window_size)
overall_users_reward_10_8_lr_smooth = moving_average(overall_users_reward_10_8_lr, window_size)
#overall_users_reward_10_5_lr_smooth = moving_average(overall_users_reward_256_steps, window_size)

energy_rewards_10_5_lr_smooth = moving_average(energy_rewards_10_5_lr, window_size)
energy_rewards_10_6_lr_smooth = moving_average(energy_rewards_10_6_lr, window_size)
energy_rewards_10_8_lr_smooth = moving_average(energy_rewards_10_8_lr, window_size)

energy_rewards_10_5_lr_smooth = energy_rewards_10_5_lr_smooth/10**3
energy_rewards_10_6_lr_smooth = energy_rewards_10_6_lr_smooth/10**3
energy_rewards_10_8_lr_smooth = energy_rewards_10_8_lr_smooth/10**3

# energy_rewards_10_6_lr_smooth = [num + random.uniform(noise_range[0], noise_range[1]) for num in energy_rewards_10_6_lr_smooth]
# energy_rewards_10_8_lr_smooth = [num + random.uniform(noise_range[0], noise_range[1]) for num in energy_rewards_10_8_lr_smooth]
#energy_rewards_256_steps_smooth = moving_average(energy_rewards_256_steps, window_size)

throughput_rewards_10_5_lr_smooth = moving_average(throughput_rewards_10_5_lr, window_size)
throughput_rewards_10_6_lr_smooth = moving_average(throughput_rewards_10_6_lr, window_size)
throughput_rewards_10_8_lr_smooth = moving_average(throughput_rewards_10_8_lr, window_size)
#throughput_rewards_256_steps_smooth = moving_average(throughput_rewards_256_steps, window_size)

delay_rewards_10_5_lr_smooth = moving_average(delay_rewards_10_5_lr, window_size)
delay_rewards_10_6_lr_smooth = moving_average(delay_rewards_10_6_lr, window_size)
delay_rewards_10_8_lr_smooth = moving_average(delay_rewards_10_8_lr, window_size)
#delay_rewards_256_steps_smooth = moving_average(delay_rewards_256_steps, window_size)
print('overall_users_reward_10_5_lr_smooth: ', overall_users_reward_10_5_lr_smooth[100])
print('throughput_rewards_10_5_lr_smooth: ', throughput_rewards_10_5_lr_smooth[100])
print('energy_rewards_10_5_lr_smooth: ', energy_rewards_10_5_lr_smooth[100])
print('delay_rewards_10_5_lr_smooth: ', delay_rewards_10_5_lr_smooth[100])
# overall_users_reward_11_users_smooth = moving_average(overall_users_reward_11_users, window_size)
# rewards_3_users_smooth = moving_average(rewards_3_users_normalized, window_size)
# rewards_5_users_smooth = moving_average(rewards_5_users_normalized, window_size)
# rewards_7_users_smooth = moving_average(rewards_7_users_normalized, window_size)
# rewards_9_users_smooth = moving_average(rewards_9_users_normalized, window_size)
#rewards_11_users_smooth = moving_average(rewards_11_users, window_size)

# len_timesteps = len(timesteps_1_users[window_size-1:])
# print(len_timesteps)

def detect_convergence_gradient(data, threshold=0.001, window_size=50):
    """
    Detect the x-axis index where convergence occurs based on gradient threshold.
    
    Parameters:
    - data: Array of rewards (or any other metric) over time.
    - threshold: Maximum allowed absolute gradient to consider convergence.
    - window_size: Number of consecutive gradients below the threshold required for convergence.
    
    Returns:
    - convergence_index: The x-axis index where convergence is detected, or None if not detected.
    """
    # Compute the gradient (absolute differences)
    gradients = np.abs(np.diff(data))
    #print('gradients: ', gradients)
    
    # Check for sustained small gradients
    for i in range(len(gradients) - window_size):
        window = gradients[i:i + window_size]
        if np.all(window < threshold):  # All gradients in the window must be below the threshold
            return i + window_size  # Return the index where the window ends
    return None  # Convergence not detected

# Detect convergence for 3, 7, and 11 users based on gradient threshold
#convergence_3_users = detect_convergence_gradient(overall_users_reward_32_steps_smooth, threshold=0.001, window_size=100)
#convergence_7_users = detect_convergence_gradient(overall_users_reward_128_steps_smooth, threshold=0.001, window_size=100)
#convergence_11_users = detect_convergence_gradient(overall_users_reward_11_users_smooth, threshold=0.001, window_size=100)

# print("Convergence for 3 Users at Episode:", convergence_3_users)
# print("Convergence for 7 Users at Episode:", convergence_7_users)
# print("Convergence for 11 Users at Episode:", convergence_11_users)


new_timesteps_10_5_lr = []
count = 0
for timestep in timesteps_10_5_lr:
    new_timesteps_10_5_lr.append(count)
    count+=1

new_timesteps_10_6_lr = []
count = 0
for timestep in timesteps_10_6_lr:
    new_timesteps_10_6_lr.append(count)
    count+=1
    
new_timesteps_10_8_lr = []
count = 0
for timestep in timesteps_10_8_lr:
    new_timesteps_10_8_lr.append(count)
    count+=1

figure, axis = plt.subplots(2,2,figsize=(10, 8))
#plt.figure(figsize=(15, 8))

title_fontsize = 15
xlabel_fontsize = 15
ylabel_fontsize = 15
legend_fontsize = 15
x_and_y_tick_fontsize = 15

axis[0,0].plot(new_timesteps_10_5_lr[window_size-1:], overall_users_reward_10_5_lr_smooth, color="green", label=r"TD3 $10^{-5}$ Actor Learning Rate")
axis[0,0].plot(new_timesteps_10_6_lr[window_size-1:], overall_users_reward_10_6_lr_smooth, color="red", label=r"TD3 $10^{-6}$ Actor Learning Rate")
axis[0,0].plot(new_timesteps_10_8_lr[window_size-1:], overall_users_reward_10_8_lr_smooth, color="purple", label=r"TD3 $10^{-8}$ Actor Learning Rate")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,0].set_title('Total System Reward',fontsize=title_fontsize, fontweight='bold')
axis[0,0].grid()
axis[0,0].set_ylabel('Reward',fontsize=ylabel_fontsize)
axis[0,0].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[0,0].legend(loc="lower right",fontsize=legend_fontsize)
axis[0,0].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[0,0].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)

axis[0,1].plot(new_timesteps_10_5_lr[window_size-1:], throughput_rewards_10_5_lr_smooth, color="green", label=r"TD3 $10^{-5}$ Actor Learning Rate")
axis[0,1].plot(new_timesteps_10_6_lr[window_size-1:], throughput_rewards_10_6_lr_smooth, color="red", label=r"TD3 $10^{-6}$ Actor Learning Rate")
axis[0,1].plot(new_timesteps_10_8_lr[window_size-1:], throughput_rewards_10_8_lr_smooth, color="purple", label=r"TD3 $10^{-8}$ Actor Learning Rate")
#axis[0,1].plot(timesteps_256_steps[window_size-1:], throughput_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[0,1].set_title('Sum Data Rate',fontsize=title_fontsize, fontweight='bold')
axis[0,1].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[0,1].set_ylabel('Data Rate (bits/s)',fontsize=ylabel_fontsize)
axis[0,1].grid()
axis[0,1].legend(loc="lower right",fontsize=legend_fontsize)
axis[0,1].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[0,1].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")

axis[1,0].plot(new_timesteps_10_5_lr[window_size-1:], energy_rewards_10_5_lr_smooth, color="green", label=r"TD3 $10^{-5}$ Actor Learning Rate")
axis[1,0].plot(new_timesteps_10_6_lr[window_size-1:], energy_rewards_10_6_lr_smooth, color="red", label=r"TD3 $10^{-6}$ Actor Learning Rate")
axis[1,0].plot(new_timesteps_10_8_lr[window_size-1:], energy_rewards_10_8_lr_smooth, color="purple", label=r"TD3 $10^{-8}$ Actor Learning Rate")
#axis[1,0].plot(timesteps_256_steps[window_size-1:], energy_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[1,0].set_title('Sum Energy Consumption',fontsize=title_fontsize, fontweight='bold')
axis[1,0].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[1,0].set_ylabel('Energy (J)',fontsize=ylabel_fontsize)
axis[1,0].grid()
axis[1,0].legend(loc="upper right",fontsize=legend_fontsize)
axis[1,0].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[1,0].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")


axis[1,1].plot(new_timesteps_10_5_lr[window_size-1:], delay_rewards_10_5_lr_smooth, color="green", label=r"TD3 $10^{-5}$ Actor Learning Rate")
axis[1,1].plot(new_timesteps_10_6_lr[window_size-1:], delay_rewards_10_6_lr_smooth, color="red", label=r"TD3 $10^{-6}$ Actor Learning Rate")
axis[1,1].plot(new_timesteps_10_8_lr[window_size-1:], delay_rewards_10_8_lr_smooth, color="purple", label=r"TD3 $10^{-8}$ Actor Learning Rate")
#axis[1,1].plot(timesteps_256_steps[window_size-1:], delay_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[1,1].set_title('Sum Delay',fontsize=title_fontsize, fontweight='bold')
axis[1,1].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[1,1].set_ylabel('Delay (ms)',fontsize=ylabel_fontsize)
axis[1,1].grid()
axis[1,1].legend(loc="upper right",fontsize=legend_fontsize)
axis[1,1].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[1,1].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")

#plt.plot(new_timesteps[window_size-1:], overall_users_reward_11_users_smooth, color="blue", label='7 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

# Mark convergence points on the main plot
# if convergence_3_users:
#     plt.axvline(convergence_3_users, color="green", linestyle="--", label="3 Users Convergence")
# if convergence_7_users:
#     plt.axvline(convergence_7_users, color="brown", linestyle="--", label="7 Users Convergence")
# if convergence_11_users:
#     plt.axvline(convergence_11_users, color="blue", linestyle="--", label="11 Users Convergence")
# plt.legend()


# plt.xlabel("Episodes")
# plt.ylabel("System Reward")
# plt.legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")
# plt.grid()
plt.xticks(fontsize=16)  # X-axis tick font size
plt.yticks(fontsize=16)  # Y-axis tick font size
plt.tight_layout()
plt.show()



