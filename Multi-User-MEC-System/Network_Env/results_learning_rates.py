import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
rewards_energy_throughput_1 = np.load('timestep_rewards_energy_throughput_1.npy')
rewards_energy_throughput_2 = np.load('timestep_rewards_energy_throughput_2.npy')
rewards_energy_throughput_3 = np.load('timestep_rewards_energy_throughput_3.npy')
rewards_energy_throughput_4 = np.load('timestep_rewards_energy_throughput_4.npy')




#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_1 = rewards_energy_throughput_1[:,0]
timesteps_2 = rewards_energy_throughput_2[:,0]
timesteps_3 = rewards_energy_throughput_3[:,0]
timesteps_4 = rewards_energy_throughput_4[:,0]

rewards_1 = rewards_energy_throughput_1[:,1]
rewards_2 = rewards_energy_throughput_2[:,1]
rewards_3 = rewards_energy_throughput_3[:,1]
rewards_4 = rewards_energy_throughput_4[:,1]



def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

rewards_1_smooth = moving_average(rewards_1, window_size)
rewards_2_smooth = moving_average(rewards_2, window_size)
rewards_3_smooth = moving_average(rewards_3, window_size)
rewards_4_smooth = moving_average(rewards_4, window_size)


plt.plot(timesteps_1[window_size-1:], rewards_1_smooth, color="green", label="0.15")
plt.plot(timesteps_2[window_size-1:], rewards_2_smooth, color="brown", label="0.35")
#plt.plot(timesteps_0_45[window_size-1:], rewards_0_45_smooth, color="brown", label='0.45')
plt.plot(timesteps_3[window_size-1:], rewards_3_smooth, color="blue", label='0.65')
plt.plot(timesteps_4[window_size-1:], rewards_4_smooth, color="red", label='0.99')
#plt.plot(timesteps_9_users[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Timestep(t)")
plt.ylabel("System Reward($\mathcal{R}$)")
plt.legend(["Alr = 10^-8, Clr = 10^-5", "Alr = 10^-8, Clr = 10^-6", "Alr = 10^-9, Clr = 10^-5", "Alr = 10^-9, Clr = 10^-6"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
