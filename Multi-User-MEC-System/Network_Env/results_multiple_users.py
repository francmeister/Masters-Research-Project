import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_1_user = np.load('timestep_rewards_energy_throughput_1_Users.npy')
rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')




#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_1_users = rewards_throughput_energy_1_user[:,0]
timesteps_3_users = rewards_throughput_energy_3_user[:,0]
timesteps_7_users = rewards_throughput_energy_7_user[:,0]
timesteps_9_users = rewards_throughput_energy_9_user[:,0]

rewards_1_users = rewards_throughput_energy_1_user[:,1]
rewards_3_users = rewards_throughput_energy_3_user[:,1]
rewards_7_users = rewards_throughput_energy_7_user[:,1]
rewards_9_users = rewards_throughput_energy_9_user[:,1]


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

rewards_1_users_smooth = moving_average(rewards_1_users, window_size)
rewards_3_users_smooth = moving_average(rewards_3_users, window_size)
rewards_7_users_smooth = moving_average(rewards_7_users, window_size)
rewards_9_users_smooth = moving_average(rewards_9_users, window_size)


plt.plot(timesteps_1_users[window_size-1:], rewards_1_users_smooth, color="green", label="1 User")
plt.plot(timesteps_3_users[window_size-1:], rewards_3_users_smooth, color="blue", label='3 Users')
plt.plot(timesteps_7_users[window_size-1:], rewards_7_users_smooth, color="red", label='7 Users')
plt.plot(timesteps_9_users[window_size-1:], rewards_9_users_smooth, color="black", label='9 Users')

plt.xlabel("Timestep(t)")
plt.ylabel("System Reward($\mathcal{R}$)")
plt.legend(["1 User","3 Users", "7 Users", "9 Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()



