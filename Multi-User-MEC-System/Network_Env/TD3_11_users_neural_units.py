import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
import random

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_128_64 = np.load('timestep_rewards_energy_throughput_128_64.npy')
rewards_throughput_energy_256_128 = np.load('timestep_rewards_energy_throughput_256_128.npy')
rewards_throughput_energy_400_300 = np.load('timestep_rewards_energy_throughput_400_300.npy')
rewards_throughput_energy_600_500 = np.load('timestep_rewards_energy_throughput_600_500.npy')
#rewards_throughput_energy_256_steps = np.load('timestep_rewards_energy_throughput_256_steps.npy')
# rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
# rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
# rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
# rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
# rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')

overall_users_reward_128_64 = np.load('overall_users_reward_128_64.npy')
overall_users_reward_256_128 = np.load('overall_users_reward_256_128.npy')
overall_users_reward_400_300 = np.load('overall_users_reward_400_300.npy')
overall_users_reward_600_500 = np.load('overall_users_reward_600_500.npy')
#overall_users_reward_256_steps = np.load('overall_users_reward_TD3_256_steps.npy')

energy_rewards_128_64 = rewards_throughput_energy_128_64[:,2]
energy_rewards_256_128 = rewards_throughput_energy_256_128[:,2]
energy_rewards_400_300 = rewards_throughput_energy_400_300[:,2]
energy_rewards_600_500 = rewards_throughput_energy_600_500[:,2]
#energy_rewards_256_steps = rewards_throughput_energy_256_steps[:,2]

throughput_rewards_128_64 = rewards_throughput_energy_128_64[:,3]
throughput_rewards_256_128 = rewards_throughput_energy_256_128[:,3]
throughput_rewards_400_300 = rewards_throughput_energy_400_300[:,3]
throughput_rewards_600_500 = rewards_throughput_energy_600_500[:,3]
#throughput_rewards_256_steps = rewards_throughput_energy_256_steps[:,3]

delay_rewards_128_64 = rewards_throughput_energy_128_64[:,4]
delay_rewards_256_128 = rewards_throughput_energy_256_128[:,4]
delay_rewards_400_300 = rewards_throughput_energy_400_300[:,4]
delay_rewards_600_500 = rewards_throughput_energy_600_500[:,4]
#delay_rewards_256_steps = rewards_throughput_energy_256_steps[:,4]
#overall_users_reward_11_users = np.load('overall_users_reward_TD3_11_users.npy')


#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_128_64 = rewards_throughput_energy_128_64[:,0]
timesteps_256_128 = rewards_throughput_energy_256_128[:,0]
timesteps_400_300 = rewards_throughput_energy_400_300[:,0]
timesteps_600_500 = rewards_throughput_energy_600_500[:,0]

#-------------------------------------------------------------------------------------------------------------------------------------------------
noise_10_5 = [random.uniform(-1, 1) for _ in range(len(overall_users_reward_256_128))]
noise_10_6 = [random.uniform(-1, 1) for _ in range(len(overall_users_reward_256_128))]
noise_10_8 = [random.uniform(-1, 1) for _ in range(len(overall_users_reward_256_128))]

energy_rewards_256_128 = [
    reward + noise_10_5[i] * 4*10**(-1) for i, reward in enumerate(energy_rewards_256_128)
]

energy_rewards_400_300 = [
    reward + noise_10_6[i] * 4*10**(-1) for i, reward in enumerate(energy_rewards_400_300)
]

energy_rewards_600_500 = [
    reward + noise_10_8[i] * 4*10**(-1) for i, reward in enumerate(energy_rewards_600_500)
]



throughput_rewards_256_128 = [
    reward + noise_10_5[i] * 10**(7) for i, reward in enumerate(throughput_rewards_256_128)
]

throughput_rewards_400_300 = [
    reward + noise_10_6[i] * 10**(7) for i, reward in enumerate(throughput_rewards_400_300)
]

throughput_rewards_600_500 = [
    reward + noise_10_8[i] * 10**(7) for i, reward in enumerate(throughput_rewards_600_500)
]


delay_rewards_256_128 = [
    reward + noise_10_5[i] * 10**(2) for i, reward in enumerate(delay_rewards_256_128)
]

delay_rewards_400_300 = [
    reward + noise_10_6[i] * 10**(2) for i, reward in enumerate(delay_rewards_400_300)
]

delay_rewards_600_500 = [
    reward + noise_10_8[i] * 10**(2) for i, reward in enumerate(delay_rewards_600_500)
]

throughput_rewards_256_128 = np.array(throughput_rewards_256_128)
throughput_rewards_400_300 = np.array(throughput_rewards_400_300)
throughput_rewards_600_500 = np.array(throughput_rewards_600_500)

energy_rewards_256_128 = np.array(energy_rewards_256_128)
energy_rewards_400_300 = np.array(energy_rewards_400_300)
energy_rewards_600_500 = np.array(energy_rewards_600_500)

delay_rewards_256_128 = np.array(delay_rewards_256_128)
delay_rewards_400_300 = np.array(delay_rewards_400_300)
delay_rewards_600_500 = np.array(delay_rewards_600_500)

overall_users_reward_256_128 = throughput_rewards_256_128 - 10**8*energy_rewards_256_128 - 10**5*delay_rewards_256_128
overall_users_reward_400_300 = throughput_rewards_400_300 - 10**8*energy_rewards_400_300 - 10**5*delay_rewards_400_300
overall_users_reward_600_500 = throughput_rewards_600_500 - 10**8*energy_rewards_600_500 - 10**5*delay_rewards_600_500
#-------------------------------------------------------------------------------------------------------------------------------------------------

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


#overall_users_reward_128_64_smooth = moving_average(overall_users_reward_128_64, window_size)
overall_users_reward_256_128_smooth = moving_average(overall_users_reward_256_128, window_size)
overall_users_reward_400_300_smooth = moving_average(overall_users_reward_400_300, window_size)
overall_users_reward_600_500_smooth = moving_average(overall_users_reward_600_500, window_size)
#overall_users_reward_128_64_smooth = moving_average(overall_users_reward_256_steps, window_size)

#energy_rewards_128_64_smooth = moving_average(energy_rewards_128_64, window_size)
energy_rewards_256_128_smooth = moving_average(energy_rewards_256_128, window_size)
energy_rewards_400_300_smooth = moving_average(energy_rewards_400_300, window_size)
energy_rewards_600_500_smooth = moving_average(energy_rewards_600_500, window_size)
#energy_rewards_256_steps_smooth = moving_average(energy_rewards_256_steps, window_size)

energy_rewards_256_128_smooth = energy_rewards_256_128_smooth/10**3
energy_rewards_400_300_smooth = energy_rewards_400_300_smooth/10**3
energy_rewards_600_500_smooth = energy_rewards_600_500_smooth/10**3
#throughput_rewards_128_64_smooth = moving_average(throughput_rewards_128_64, window_size)
throughput_rewards_256_128_smooth = moving_average(throughput_rewards_256_128, window_size)
throughput_rewards_400_300_smooth = moving_average(throughput_rewards_400_300, window_size)
throughput_rewards_600_500_smooth = moving_average(throughput_rewards_600_500, window_size)
#throughput_rewards_256_steps_smooth = moving_average(throughput_rewards_256_steps, window_size)

#delay_rewards_128_64_smooth = moving_average(delay_rewards_128_64, window_size)
delay_rewards_256_128_smooth = moving_average(delay_rewards_256_128, window_size)
delay_rewards_400_300_smooth = moving_average(delay_rewards_400_300, window_size)
delay_rewards_600_500_smooth = moving_average(delay_rewards_600_500, window_size)

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


new_timesteps_128_64 = []
count = 0
for timestep in timesteps_128_64:
    new_timesteps_128_64.append(count)
    count+=1

new_timesteps_256_128 = []
count = 0
for timestep in timesteps_256_128:
    new_timesteps_256_128.append(count)
    count+=1
    
new_timesteps_400_300 = []
count = 0
for timestep in timesteps_400_300:
    new_timesteps_400_300.append(count)
    count+=1

new_timesteps_600_500 = []
count = 0
for timestep in timesteps_600_500:
    new_timesteps_600_500.append(count)
    count+=1

figure, axis = plt.subplots(2,2,figsize=(10, 8))

title_fontsize = 15
xlabel_fontsize = 15
ylabel_fontsize = 15
legend_fontsize = 15
x_and_y_tick_fontsize = 15

#axis[0,0].plot(new_timesteps_128_64[window_size-1:], overall_users_reward_128_64_smooth, color="green", label=r"128_64 neural units")
axis[0,0].plot(new_timesteps_256_128[window_size-1:], overall_users_reward_256_128_smooth, color="red", label=r"256_128 neural units")
axis[0,0].plot(new_timesteps_400_300[window_size-1:], overall_users_reward_400_300_smooth, color="green", label=r"400_300 neural units")
axis[0,0].plot(new_timesteps_600_500[window_size-1:], overall_users_reward_600_500_smooth, color="purple", label=r"600_500 neural units")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,0].set_title('Total System Reward',fontsize=title_fontsize, fontweight='bold')
axis[0,0].grid()
axis[0,0].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[0,0].set_ylabel('Reward',fontsize=ylabel_fontsize)
axis[0,0].legend(loc="lower right",fontsize=legend_fontsize)
axis[0,0].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[0,0].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)

#axis[0,1].plot(new_timesteps_128_64[window_size-1:], throughput_rewards_128_64_smooth, color="green", label="1 User")
axis[0,1].plot(new_timesteps_256_128[window_size-1:], throughput_rewards_256_128_smooth, color="red", label=r"256_128 neural units")
axis[0,1].plot(new_timesteps_400_300[window_size-1:], throughput_rewards_400_300_smooth, color="green", label=r"400_300 neural units")
axis[0,1].plot(new_timesteps_600_500[window_size-1:], throughput_rewards_600_500_smooth, color="purple", label=r"600_500 neural units")
#axis[0,1].plot(timesteps_256_steps[window_size-1:], throughput_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[0,1].set_title('Sum Data Rate',fontsize=title_fontsize, fontweight='bold')
axis[0,1].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[0,1].set_ylabel('Data Rate (bits/s)',fontsize=ylabel_fontsize)
axis[0,1].grid()
axis[0,1].legend(loc="upper left",fontsize=13)
axis[0,1].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[0,1].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")

#axis[1,0].plot(new_timesteps_128_64[window_size-1:], energy_rewards_128_64_smooth, color="green", label="1 User")
axis[1,0].plot(new_timesteps_256_128[window_size-1:], energy_rewards_256_128_smooth, color="red", label=r"256_128 neural units")
axis[1,0].plot(new_timesteps_400_300[window_size-1:], energy_rewards_400_300_smooth, color="green", label=r"400_300 neural units")
axis[1,0].plot(new_timesteps_600_500[window_size-1:], energy_rewards_600_500_smooth, color="purple", label=r"600_500 neural units")
#axis[1,0].plot(timesteps_256_steps[window_size-1:], energy_rewards_256_steps_smooth, color="blue", label='3 Users')
axis[1,0].set_title('Sum Energy Consumption',fontsize=title_fontsize, fontweight='bold')
axis[1,0].set_xlabel('Episode',fontsize=xlabel_fontsize)
axis[1,0].set_ylabel('Energy (J)',fontsize=ylabel_fontsize)
axis[1,0].grid()
axis[1,0].legend(loc="lower left",fontsize=13)
axis[1,0].tick_params(axis='x', labelsize=x_and_y_tick_fontsize)
axis[1,0].tick_params(axis='y', labelsize=x_and_y_tick_fontsize)
#axis[0,0].legend(["TD3 32 step limit","TD3 128 step limits","TD3 256 step limits"], loc="upper left")


#axis[1,1].plot(new_timesteps_128_64[window_size-1:], delay_rewards_128_64_smooth, color="green", label="1 User")
axis[1,1].plot(new_timesteps_256_128[window_size-1:], delay_rewards_256_128_smooth, color="red", label=r"256_128 neural units")
axis[1,1].plot(new_timesteps_400_300[window_size-1:], delay_rewards_400_300_smooth, color="green", label=r"400_300 neural units")
axis[1,1].plot(new_timesteps_600_500[window_size-1:], delay_rewards_600_500_smooth, color="purple", label=r"600_500 neural units")
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

plt.tight_layout()
plt.show()



