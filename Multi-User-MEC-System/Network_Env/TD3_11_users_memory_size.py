import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_10_4 = np.load('timestep_rewards_energy_throughput_10_4.npy')
rewards_throughput_energy_10_5 = np.load('timestep_rewards_energy_throughput_10_5.npy')
rewards_throughput_energy_10_6 = np.load('timestep_rewards_energy_throughput_10_6.npy')
#rewards_throughput_energy_256_steps = np.load('timestep_rewards_energy_throughput_256_steps.npy')
# rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
# rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
# rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
# rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
# rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')

overall_users_reward_10_4 = np.load('overall_users_reward_10_4.npy')
overall_users_reward_10_5 = np.load('overall_users_reward_10_5.npy')
overall_users_reward_10_6 = np.load('overall_users_reward_10_6.npy')
#overall_users_reward_256_steps = np.load('overall_users_reward_TD3_256_steps.npy')

energy_rewards_10_4 = rewards_throughput_energy_10_4[:,2]
energy_rewards_10_5 = rewards_throughput_energy_10_5[:,2]
energy_rewards_10_6 = rewards_throughput_energy_10_6[:,2]
#energy_rewards_256_steps = rewards_throughput_energy_256_steps[:,2]

throughput_rewards_10_4 = rewards_throughput_energy_10_4[:,3]
throughput_rewards_10_5 = rewards_throughput_energy_10_5[:,3]
throughput_rewards_10_6 = rewards_throughput_energy_10_6[:,3]
#throughput_rewards_256_steps = rewards_throughput_energy_256_steps[:,3]

delay_rewards_10_4 = rewards_throughput_energy_10_4[:,4]
delay_rewards_10_5 = rewards_throughput_energy_10_5[:,4]
delay_rewards_10_6 = rewards_throughput_energy_10_6[:,4]
#delay_rewards_256_steps = rewards_throughput_energy_256_steps[:,4]
#overall_users_reward_11_users = np.load('overall_users_reward_TD3_11_users.npy')


#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_10_4 = rewards_throughput_energy_10_4[:,0]
timesteps_10_5 = rewards_throughput_energy_10_5[:,0]
timesteps_10_6 = rewards_throughput_energy_10_6[:,0]
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


overall_users_reward_10_4_smooth = moving_average(overall_users_reward_10_4, window_size)
overall_users_reward_10_5_smooth = moving_average(overall_users_reward_10_5, window_size)
overall_users_reward_10_6_smooth = moving_average(overall_users_reward_10_6, window_size)
#overall_users_reward_10_4_smooth = moving_average(overall_users_reward_256_steps, window_size)

energy_rewards_10_4_smooth = moving_average(energy_rewards_10_4, window_size)
energy_rewards_10_5_smooth = moving_average(energy_rewards_10_5, window_size)
energy_rewards_10_6_smooth = moving_average(energy_rewards_10_6, window_size)
#energy_rewards_256_steps_smooth = moving_average(energy_rewards_256_steps, window_size)

throughput_rewards_10_4_smooth = moving_average(throughput_rewards_10_4, window_size)
throughput_rewards_10_5_smooth = moving_average(throughput_rewards_10_5, window_size)
throughput_rewards_10_6_smooth = moving_average(throughput_rewards_10_6, window_size)
#throughput_rewards_256_steps_smooth = moving_average(throughput_rewards_256_steps, window_size)

delay_rewards_10_4_smooth = moving_average(delay_rewards_10_4, window_size)
delay_rewards_10_5_smooth = moving_average(delay_rewards_10_5, window_size)
delay_rewards_10_6_smooth = moving_average(delay_rewards_10_6, window_size)
#delay_rewards_256_steps_smooth = moving_average(delay_rewards_256_steps, window_size)

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


new_timesteps_10_4 = []
count = 0
for timestep in timesteps_10_4:
    new_timesteps_10_4.append(count)
    count+=1

new_timesteps_10_5 = []
count = 0
for timestep in timesteps_10_5:
    new_timesteps_10_5.append(count)
    count+=1
    
new_timesteps_10_6 = []
count = 0
for timestep in timesteps_10_6:
    new_timesteps_10_6.append(count)
    count+=1

figure, axis = plt.subplots(2,2)

axis[0,0].plot(new_timesteps_10_4[window_size-1:], overall_users_reward_10_4_smooth, color="green", label=r"TD3 $10^{4}$ Memory Size")
axis[0,0].plot(new_timesteps_10_5[window_size-1:], overall_users_reward_10_5_smooth, color="red", label=r"TD3 $10^{5}$ Memory Size")
axis[0,0].plot(new_timesteps_10_6[window_size-1:], overall_users_reward_10_6_smooth, color="brown", label=r"TD3 $10^{6}$ Memory Size")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,0].set_title('Total System Reward')
axis[0,0].grid()
axis[0,0].set_xlabel('Episode')
axis[0,0].legend(loc="lower right")

axis[0,1].plot(new_timesteps_10_4[window_size-1:], throughput_rewards_10_4_smooth, color="green", label="1 User")
axis[0,1].plot(new_timesteps_10_5[window_size-1:], throughput_rewards_10_5_smooth, color="red", label="1 User")
axis[0,1].plot(new_timesteps_10_6[window_size-1:], throughput_rewards_10_6_smooth, color="brown", label='3 Users')
#axis[0,1].plot(timesteps_256_steps[window_size-1:], throughput_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[0,1].set_title('Sum Data Rates')
axis[0,1].set_xlabel('Episode')
axis[0,1].set_ylabel('Data Rate (bits/s)')
axis[0,1].grid()
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")

axis[1,0].plot(new_timesteps_10_4[window_size-1:], energy_rewards_10_4_smooth, color="green", label="1 User")
axis[1,0].plot(new_timesteps_10_5[window_size-1:], energy_rewards_10_5_smooth, color="red", label="1 User")
axis[1,0].plot(new_timesteps_10_6[window_size-1:], energy_rewards_10_6_smooth, color="brown", label='3 Users')
#axis[1,0].plot(timesteps_256_steps[window_size-1:], energy_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[1,0].set_title('Energy Consumption')
axis[1,0].set_xlabel('Episode')
axis[1,0].set_ylabel('Energy (J)')
axis[1,0].grid()
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")


axis[1,1].plot(new_timesteps_10_4[window_size-1:], delay_rewards_10_4_smooth, color="green", label="1 User")
axis[1,1].plot(new_timesteps_10_5[window_size-1:], delay_rewards_10_5_smooth, color="red", label="1 User")
axis[1,1].plot(new_timesteps_10_6[window_size-1:], delay_rewards_10_6_smooth, color="brown", label='3 Users')
#axis[1,1].plot(timesteps_256_steps[window_size-1:], delay_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[1,1].set_title('Sum Delay')
axis[1,1].set_xlabel('Episode')
axis[1,1].set_ylabel('Delay (ms)')
axis[1,1].grid()
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

plt.tight_layout()
plt.show()



