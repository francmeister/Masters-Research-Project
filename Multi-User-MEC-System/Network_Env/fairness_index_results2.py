import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


timestep_rewards_energy_throughput_0_1 = np.load('timestep_rewards_energy_throughput_0_1.npy')
timestep_rewards_energy_throughput_0_2 = np.load('timestep_rewards_energy_throughput_0_2.npy')
timestep_rewards_energy_throughput_0_4 = np.load('timestep_rewards_energy_throughput_0_4.npy')
timestep_rewards_energy_throughput_0_8 = np.load('timestep_rewards_energy_throughput_0_8.npy')
timestep_rewards_energy_throughput_1 = np.load('timestep_rewards_energy_throughput_1.npy')


fairness_index_0_1 = np.load('fairnes_index_0_1.npy') 
fairness_index_0_2 = np.load('fairnes_index_0_2.npy') 
fairness_index_0_4 = np.load('fairnes_index_0_4.npy') 
fairness_index_0_8 = np.load('fairnes_index_0_8.npy') 
fairness_index_1 = np.load('fairnes_index_1.npy') 

timesteps_0_1 = timestep_rewards_energy_throughput_0_1[:,0]
timesteps_0_2 = timestep_rewards_energy_throughput_0_2[:,0]
timesteps_0_4 = timestep_rewards_energy_throughput_0_4[:,0]
timesteps_0_8 = timestep_rewards_energy_throughput_0_8[:,0]
timesteps_1 = timestep_rewards_energy_throughput_1[:,0]

rewards_0_1 = timestep_rewards_energy_throughput_0_1[:,1]
rewards_0_2 = timestep_rewards_energy_throughput_0_2[:,1]
rewards_0_4 = timestep_rewards_energy_throughput_0_4[:,1]
rewards_0_8 = timestep_rewards_energy_throughput_0_8[:,1]
rewards_1 = timestep_rewards_energy_throughput_1[:,1]




def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 1000

rewards_0_1_smooth = moving_average(rewards_0_1, window_size)
rewards_0_2_smooth = moving_average(rewards_0_2, window_size)
rewards_0_4_smooth = moving_average(rewards_0_4, window_size)
rewards_0_8_smooth = moving_average(rewards_0_8, window_size)
rewards_1_smooth = moving_average(rewards_1, window_size)

fairness_index_0_1_smooth = moving_average(fairness_index_0_1, window_size)
fairness_index_0_2_smooth = moving_average(fairness_index_0_2, window_size)
fairness_index_0_4_smooth = moving_average(fairness_index_0_4, window_size)
fairness_index_0_8_smooth = moving_average(fairness_index_0_8, window_size)
fairness_index_1_smooth = moving_average(fairness_index_1, window_size)


# plt.plot(timesteps_0_1[window_size-1:], rewards_0_1_smooth, color="green", label="1 User")
# plt.plot(timesteps_0_2[window_size-1:], rewards_0_2_smooth, color="blue", label='3 Users')
# plt.plot(timesteps_0_4[window_size-1:], rewards_0_4_smooth, color="red", label='5 Users')
# plt.plot(timesteps_0_8[window_size-1:], rewards_0_8_smooth, color="purple", label='7 Users')
# plt.plot(timesteps_1[window_size-1:], rewards_1_smooth, color="grey", label='9 Users')

figure, axis = plt.subplots(2,1)

axis[0].plot(timesteps_0_1[window_size-1:], rewards_0_1_smooth, color="green")
axis[0].plot(timesteps_0_2[window_size-1:], rewards_0_2_smooth, color="blue")
axis[0].plot(timesteps_0_4[window_size-1:], rewards_0_4_smooth, color="red")
axis[0].plot(timesteps_0_8[window_size-1:], rewards_0_8_smooth, color="purple")
axis[0].plot(timesteps_1[window_size-1:], rewards_1_smooth, color="grey")
axis[0].legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")
#axis[0].set_title('DDPG Reward DF = 0.99')

axis[1].plot(timesteps_0_1[window_size-1:], fairness_index_0_1_smooth, color="green")
axis[1].plot(timesteps_0_2[window_size-1:], fairness_index_0_2_smooth, color="blue")
axis[1].plot(timesteps_0_4[window_size-1:], fairness_index_0_4_smooth, color="red")
axis[1].plot(timesteps_0_8[window_size-1:], fairness_index_0_8_smooth, color="purple")
axis[1].plot(timesteps_1[window_size-1:], fairness_index_1_smooth, color="grey")
axis[1].legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")

plt.xlabel("Timestep(t)")
plt.ylabel("Reward")
#plt.legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
