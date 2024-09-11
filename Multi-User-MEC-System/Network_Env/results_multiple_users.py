import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_1_user = np.load('timestep_rewards_energy_throughput_1_Users.npy')
rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')




#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_1_users = rewards_throughput_energy_1_user[:,0]
timesteps_3_users = rewards_throughput_energy_3_user[:,0]
timesteps_5_users = rewards_throughput_energy_5_user[:,0]
timesteps_7_users = rewards_throughput_energy_7_user[:,0]
timesteps_9_users = rewards_throughput_energy_9_user[:,0]
timesteps_11_users = rewards_throughput_energy_11_user[:,0]

rewards_1_users = rewards_throughput_energy_1_user[:,1]
rewards_3_users = rewards_throughput_energy_3_user[:,1]
rewards_5_users = rewards_throughput_energy_5_user[:,1]
rewards_7_users = rewards_throughput_energy_7_user[:,1]
rewards_9_users = rewards_throughput_energy_9_user[:,1]
rewards_11_users = rewards_throughput_energy_11_user[:,1]


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

for x in rewards_1_users:
    rewards_1_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

for x in rewards_3_users:
    rewards_3_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

for x in rewards_5_users:
    rewards_5_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

for x in rewards_7_users:
    rewards_7_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))

for x in rewards_9_users:
    rewards_9_users_normalized.append(interp(x,[0,max(rewards_9_users)],[0,1]))


rewards_1_users_smooth = moving_average(rewards_1_users_normalized, window_size)
rewards_3_users_smooth = moving_average(rewards_3_users_normalized, window_size)
rewards_5_users_smooth = moving_average(rewards_5_users_normalized, window_size)
rewards_7_users_smooth = moving_average(rewards_7_users_normalized, window_size)
rewards_9_users_smooth = moving_average(rewards_9_users_normalized, window_size)
#rewards_11_users_smooth = moving_average(rewards_11_users, window_size)

len_timesteps = len(timesteps_1_users[window_size-1:])
print(len_timesteps)

new_timesteps = []
count = 0
for timestep in timesteps_1_users:
    new_timesteps.append(count)
    count+=1


plt.plot(new_timesteps[window_size-1:], rewards_1_users_smooth, color="green", label="1 User")
plt.plot(new_timesteps[window_size-1:], rewards_7_users_smooth[0:len_timesteps], color="brown", label='3 Users')
plt.plot(new_timesteps[window_size-1:], rewards_3_users_smooth[0:len_timesteps], color="blue", label='7 Users')
plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Episodes")
plt.ylabel("Normalized System Reward")
plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users"], loc="upper left")
plt.grid()

plt.tight_layout()
plt.show()



