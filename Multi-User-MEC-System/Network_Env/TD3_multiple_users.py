import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_TD3_3_users = np.load('timestep_rewards_energy_throughput_TD3_3_users.npy')
rewards_throughput_energy_TD3_7_users = np.load('timestep_rewards_energy_throughput_TD3_7_users.npy')
rewards_throughput_energy_TD3_11_users = np.load('timestep_rewards_energy_throughput_TD3_11_users.npy')
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
timesteps_TD3_3_users = rewards_throughput_energy_TD3_3_users[:,0]
timesteps_TD3_7_users = rewards_throughput_energy_TD3_7_users[:,0]
timesteps_TD3_11_users = rewards_throughput_energy_TD3_11_users[:,0]

energy_TD3_3_users = rewards_throughput_energy_TD3_3_users[:,2]
eneryg_TD3_7_users = rewards_throughput_energy_TD3_7_users[:,2]
energy_TD3_11_users = rewards_throughput_energy_TD3_11_users[:,2]

throughput_TD3_3_users = rewards_throughput_energy_TD3_3_users[:,3]
throughput_TD3_7_users = rewards_throughput_energy_TD3_7_users[:,3]
throughput_TD3_11_users = rewards_throughput_energy_TD3_11_users[:,3]

delay_TD3_3_users = rewards_throughput_energy_TD3_3_users[:,4]
delay_TD3_7_users = rewards_throughput_energy_TD3_7_users[:,4]
delay_TD3_11_users = rewards_throughput_energy_TD3_11_users[:,4]
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


energy_TD3_3_users_smooth = moving_average(energy_TD3_3_users, window_size)
energy_TD3_7_users_smooth = moving_average(eneryg_TD3_7_users, window_size)
energy_TD3_11_users_smooth = moving_average(energy_TD3_11_users, window_size)

throughput_TD3_3_users_smooth = moving_average(throughput_TD3_3_users, window_size)
throughput_TD3_7_users_smooth = moving_average(throughput_TD3_7_users, window_size)
throughput_TD3_11_users_smooth = moving_average(throughput_TD3_11_users, window_size)

delay_TD3_3_users_smooth = moving_average(delay_TD3_3_users, window_size)
delay_TD3_7_users_smooth = moving_average(delay_TD3_7_users, window_size)
delay_TD3_11_users_smooth = moving_average(delay_TD3_11_users, window_size)
# rewards_3_users_smooth = moving_average(rewards_3_users_normalized, window_size)
# rewards_5_users_smooth = moving_average(rewards_5_users_normalized, window_size)
# rewards_7_users_smooth = moving_average(rewards_7_users_normalized, window_size)
# rewards_9_users_smooth = moving_average(rewards_9_users_normalized, window_size)
#rewards_11_users_smooth = moving_average(rewards_11_users, window_size)

# len_timesteps = len(timesteps_1_users[window_size-1:])
# print(len_timesteps)


new_timesteps_3_users = []
count = 0
for timestep in timesteps_TD3_3_users:
    new_timesteps_3_users.append(count)
    count+=1

new_timesteps_7_users = []
count = 0
for timestep in timesteps_TD3_7_users:
    new_timesteps_7_users.append(count)
    count+=1

new_timesteps_11_users = []
count = 0
for timestep in timesteps_TD3_11_users:
    new_timesteps_11_users.append(count)
    count+=1

figure, axis = plt.subplots(2,2)

axis[0,0].plot(new_timesteps_3_users[window_size-1:], overall_users_reward_3_users_smooth, color="green", label="1 User")
axis[0,0].plot(new_timesteps_7_users[window_size-1:], overall_users_reward_7_users_smooth, color="brown", label='3 Users')
axis[0,0].plot(new_timesteps_11_users[window_size-1:], overall_users_reward_11_users_smooth, color="blue", label='7 Users')
axis[0,0].set_title('Total System Reward')
axis[0,0].set_xlabel('Episode')
axis[0,0].grid()
axis[0,0].legend(["3 Users","7 Users","11 Users"], loc="lower right")


axis[0,1].plot(new_timesteps_3_users[window_size-1:], throughput_TD3_3_users_smooth, color="green", label="1 User")
axis[0,1].plot(new_timesteps_7_users[window_size-1:], throughput_TD3_7_users_smooth, color="brown", label='3 Users')
axis[0,1].plot(new_timesteps_11_users[window_size-1:], throughput_TD3_11_users_smooth, color="blue", label='7 Users')
axis[0,1].set_title('Sum Data Rates')
axis[0,1].set_xlabel('Episode')
axis[0,1].set_ylabel('Data Rate (bits/s)')
axis[0,1].grid()
#axis[0,0].legend(["3 Users","7 Users","11 Users"], loc="lower right")

axis[1,0].plot(new_timesteps_3_users[window_size-1:], energy_TD3_3_users_smooth, color="green", label="1 User")
axis[1,0].plot(new_timesteps_7_users[window_size-1:], energy_TD3_7_users_smooth, color="brown", label='3 Users')
axis[1,0].plot(new_timesteps_11_users[window_size-1:], energy_TD3_11_users_smooth, color="blue", label='7 Users')
axis[1,0].set_title('Energy Consumption')
axis[1,0].set_xlabel('Episode')
axis[1,0].set_ylabel('Energy (J)')
axis[1,0].grid()

axis[1,1].plot(new_timesteps_3_users[window_size-1:], delay_TD3_3_users_smooth, color="green", label="1 User")
axis[1,1].plot(new_timesteps_7_users[window_size-1:], delay_TD3_7_users_smooth, color="brown", label='3 Users')
axis[1,1].plot(new_timesteps_11_users[window_size-1:], delay_TD3_11_users_smooth, color="blue", label='7 Users')
axis[1,1].set_title('Sum Delay')
axis[1,1].set_xlabel('Episode')
axis[1,1].set_ylabel('Delay (ms)')
axis[1,1].grid()
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')




# plt.xlabel("Episodes")
# plt.ylabel("System Reward")
# plt.legend(["TD3 3 Users","TD3 5 Users", "TD3 7 Users"], loc="upper left")
# plt.grid()

plt.tight_layout()
plt.show()



