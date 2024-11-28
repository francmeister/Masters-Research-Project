import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')
# rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
# rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
# rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
# rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
# rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')

overall_users_reward_3_users = np.load('overall_users_reward_TD3_3_users.npy')
overall_users_reward_7_users = np.load('overall_users_reward_TD3_7_users.npy')
overall_users_reward_11_users = np.load('overall_users_reward_TD3_11_users.npy')


#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps = rewards_throughput_energy[:,0]
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


overall_users_reward_3_users_smooth = moving_average(overall_users_reward_3_users, window_size)
overall_users_reward_7_users_smooth = moving_average(overall_users_reward_7_users, window_size)
overall_users_reward_11_users_smooth = moving_average(overall_users_reward_11_users, window_size)
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
convergence_3_users = detect_convergence_gradient(overall_users_reward_3_users_smooth, threshold=0.001, window_size=100)
convergence_7_users = detect_convergence_gradient(overall_users_reward_7_users_smooth, threshold=0.001, window_size=100)
convergence_11_users = detect_convergence_gradient(overall_users_reward_11_users_smooth, threshold=0.001, window_size=100)

print("Convergence for 3 Users at Episode:", convergence_3_users)
print("Convergence for 7 Users at Episode:", convergence_7_users)
print("Convergence for 11 Users at Episode:", convergence_11_users)


new_timesteps = []
count = 0
for timestep in timesteps:
    new_timesteps.append(count)
    count+=1


plt.plot(new_timesteps[window_size-1:], overall_users_reward_3_users_smooth, color="green", label="1 User")
plt.plot(new_timesteps[window_size-1:], overall_users_reward_7_users_smooth, color="brown", label='3 Users')
plt.plot(new_timesteps[window_size-1:], overall_users_reward_11_users_smooth, color="blue", label='7 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

# Mark convergence points on the main plot
if convergence_3_users:
    plt.axvline(convergence_3_users, color="green", linestyle="--", label="3 Users Convergence")
if convergence_7_users:
    plt.axvline(convergence_7_users, color="brown", linestyle="--", label="7 Users Convergence")
if convergence_11_users:
    plt.axvline(convergence_11_users, color="blue", linestyle="--", label="11 Users Convergence")
plt.legend()


plt.xlabel("Episodes")
plt.ylabel("System Reward")
plt.legend(["TD3 3 Users","TD3 5 Users", "TD3 7 Users"], loc="upper left")
plt.grid()

plt.tight_layout()
plt.show()



