import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


fairnes_index_1_Users = np.load('fairnes_index_1_Users.npy')
fairnes_index_3_Users = np.load('fairnes_index_3_Users.npy')
fairnes_index_5_Users = np.load('fairnes_index_5_Users.npy')
fairnes_index_7_Users = np.load('fairnes_index_7_Users.npy')
fairnes_index_9_Users = np.load('fairnes_index_9_Users.npy')

rewards_throughput_energy_1_user = np.load('timestep_rewards_energy_throughput_1_Users.npy')
rewards_throughput_energy_3_user = np.load('timestep_rewards_energy_throughput_3_Users.npy')
rewards_throughput_energy_5_user = np.load('timestep_rewards_energy_throughput_5_Users.npy')
rewards_throughput_energy_7_user = np.load('timestep_rewards_energy_throughput_7_Users.npy')
rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')

timesteps_1_users = rewards_throughput_energy_1_user[:,0]
timesteps_3_users = rewards_throughput_energy_3_user[:,0]
timesteps_5_users = rewards_throughput_energy_5_user[:,0]
timesteps_7_users = rewards_throughput_energy_7_user[:,0]
timesteps_9_users = rewards_throughput_energy_9_user[:,0]



def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 1000

fairnes_index_1_Users_smooth = moving_average(fairnes_index_1_Users, window_size)
fairnes_index_3_Users_smooth = moving_average(fairnes_index_3_Users, window_size)
fairnes_index_5_Users_smooth = moving_average(fairnes_index_5_Users, window_size)
fairnes_index_7_Users_smooth = moving_average(fairnes_index_7_Users, window_size)
fairnes_index_9_Users_smooth = moving_average(fairnes_index_9_Users, window_size)

plt.plot(timesteps_1_users[window_size-1:], fairnes_index_1_Users_smooth, color="green", label="1 User")
plt.plot(timesteps_3_users[window_size-1:], fairnes_index_3_Users_smooth, color="blue", label='3 Users')
plt.plot(timesteps_5_users[window_size-1:], fairnes_index_5_Users_smooth, color="red", label='5 Users')
plt.plot(timesteps_7_users[window_size-1:], fairnes_index_7_Users_smooth, color="purple", label='7 Users')
plt.plot(timesteps_9_users[window_size-1:], fairnes_index_9_Users_smooth, color="black", label='9 Users')


plt.xlabel("Timestep(t)")
plt.ylabel("Fairness Index")
plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
