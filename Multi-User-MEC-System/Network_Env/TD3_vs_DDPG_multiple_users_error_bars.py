import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

# rewards_throughput_energy_1_user = np.load('timestep_rewards_energy_throughput_1_Users.npy')
# rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
# rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
# rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
# rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
# rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')


overall_users_reward_TD3_3_users = np.load('overall_users_reward_TD3_3_users.npy')
overall_users_reward_TD3_7_users = np.load('overall_users_reward_TD3_7_users.npy')
overall_users_reward_TD3_11_users = np.load('overall_users_reward_TD3_11_users.npy')

overall_users_reward_DDPG_3_users = np.load('overall_users_reward_DDPG_3_users.npy')
overall_users_reward_DDPG_7_users = np.load('overall_users_reward_DDPG_7_users.npy')
overall_users_reward_DDPG_11_users = np.load('overall_users_reward_DDPG_11_users.npy')

rewards_throughput_energy = np.load('timestep_rewards_energy_throughput_TD3_3_users.npy')

#print('rewards_throughput_energy: ', rewards_throughput_energy)
# timesteps_1_users = rewards_throughput_energy_1_user[:,0]
timesteps = rewards_throughput_energy[:,0]
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


overall_users_reward_TD3_3_users_smooth = moving_average(overall_users_reward_TD3_3_users, window_size)
overall_users_reward_TD3_7_users_smooth = moving_average(overall_users_reward_TD3_7_users, window_size)
overall_users_reward_TD3_11_users_smooth = moving_average(overall_users_reward_TD3_11_users, window_size)


overall_users_reward_DDPG_3_users_smooth = moving_average(overall_users_reward_DDPG_3_users, window_size)
overall_users_reward_DDPG_7_users_smooth = moving_average(overall_users_reward_DDPG_7_users, window_size)
overall_users_reward_DDPG_11_users_smooth = moving_average(overall_users_reward_DDPG_11_users, window_size)
# rewards_3_users_smooth = moving_average(rewards_3_users_normalized, window_size)
# rewards_5_users_smooth = moving_average(rewards_5_users_normalized, window_size)
# rewards_7_users_smooth = moving_average(rewards_7_users_normalized, window_size)
# rewards_9_users_smooth = moving_average(rewards_9_users_normalized, window_size)
#rewards_11_users_smooth = moving_average(rewards_11_users, window_size)

# len_timesteps = len(timesteps_1_users[window_size-1:])
# print(len_timesteps)

new_timesteps = []
count = 0
for timestep in timesteps:
    new_timesteps.append(count)
    count+=1

# Variance calculation function
def calculate_variance(data):
    mean = np.mean(data)
    variance = np.sum((data - mean) ** 2) / len(data)
    return variance

# Calculate error (standard deviation = sqrt(variance))
error_3_users_TD3 = np.sqrt(calculate_variance(overall_users_reward_TD3_3_users_smooth))
error_7_users_TD3 = np.sqrt(calculate_variance(overall_users_reward_TD3_7_users_smooth))
error_11_users_TD3 = np.sqrt(calculate_variance(overall_users_reward_TD3_11_users_smooth))

error_3_users_DDPG = np.sqrt(calculate_variance(overall_users_reward_DDPG_3_users_smooth))
error_7_users_DDPG = np.sqrt(calculate_variance(overall_users_reward_DDPG_7_users_smooth))
error_11_users_DDPG = np.sqrt(calculate_variance(overall_users_reward_DDPG_11_users_smooth))

#print('error_3_users: ', error_3_users)
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

# Main plot
axs[0].plot(timesteps[window_size-1:], overall_users_reward_TD3_3_users_smooth, color="green", label="TD3")
axs[0].plot(timesteps[window_size-1:], overall_users_reward_DDPG_3_users_smooth, color="brown", label='DDPG')

axs[0].plot(timesteps[window_size-1:], overall_users_reward_TD3_7_users_smooth, color="blue", label="TD3")
axs[0].plot(timesteps[window_size-1:], overall_users_reward_DDPG_7_users_smooth, color="yellow", label='DDPG')

axs[0].plot(timesteps[window_size-1:], overall_users_reward_TD3_11_users_smooth, color="red", label="TD3")
axs[0].plot(timesteps[window_size-1:], overall_users_reward_DDPG_11_users_smooth, color="pink", label='DDPG')

axs[0].set_xlabel("Episodes")
axs[0].set_ylabel("System Reward")
axs[0].legend(loc="upper left")
axs[0].grid()
axs[0].set_title("System Reward")

# Error plot
error_means = [np.mean(overall_users_reward_TD3_3_users_smooth),
               np.mean(overall_users_reward_DDPG_3_users_smooth),

               np.mean(overall_users_reward_TD3_7_users_smooth),
               np.mean(overall_users_reward_DDPG_7_users_smooth),

               np.mean(overall_users_reward_TD3_11_users_smooth),
               np.mean(overall_users_reward_DDPG_11_users_smooth)]

error_bars = [error_3_users_TD3, error_3_users_DDPG,error_7_users_TD3, error_7_users_DDPG, error_11_users_TD3, error_11_users_DDPG]
user_labels = ['3 Users TD3', '3 Users DDPG','7 Users TD3', '7 Users DDPG','11 Users TD3', '11 Users DDPG']

# Plot error bars with dots, using respective colors
axs[1].errorbar(user_labels, error_means, yerr=error_bars, fmt='o', 
                color='green', ecolor='green', elinewidth=1, capsize=5, markersize=6, label="3 Users TD3")
axs[1].errorbar(user_labels[1], error_means[1], yerr=error_bars[1], fmt='o', 
                color='brown', ecolor='brown', elinewidth=1, capsize=5, markersize=6, label="3 Users DDPG")
axs[1].errorbar(user_labels[2], error_means[2], yerr=error_bars[2], fmt='o', 
                color='blue', ecolor='blue', elinewidth=1, capsize=5, markersize=6, label="7 Users TD3")
axs[1].errorbar(user_labels[3], error_means[3], yerr=error_bars[3], fmt='o', 
                color='yellow', ecolor='yellow', elinewidth=1, capsize=5, markersize=6, label="7 Users DDPG")
axs[1].errorbar(user_labels[4], error_means[4], yerr=error_bars[4], fmt='o', 
                color='red', ecolor='red', elinewidth=1, capsize=5, markersize=6, label="11 Users TD3")
axs[1].errorbar(user_labels[5], error_means[5], yerr=error_bars[5], fmt='o', 
                color='pink', ecolor='pink', elinewidth=1, capsize=5, markersize=6, label="11 Users DDPG")

axs[1].set_ylabel("Mean Reward")
axs[1].set_title("Mean Reward with Error Bars")
axs[1].grid(axis='y')

# Rotate x-axis labels by 45 degrees
axs[1].set_xticks(range(len(user_labels)))  # Ensure correct ticks for labels
axs[1].set_xticklabels(user_labels, rotation=45, ha="right")

plt.tight_layout()
plt.show()




