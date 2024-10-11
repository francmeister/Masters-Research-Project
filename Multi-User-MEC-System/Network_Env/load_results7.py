import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
import math

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
F_L_inverse = np.load('F_L_inverse.npy')
urllc_total_rate = np.load('urllc_total_rate.npy')
rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')


timesteps = rewards_throughput_energy[:,0]

def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

figure, axis = plt.subplots(1,2)

F_L_inverse_smooth = moving_average(F_L_inverse, window_size)
urllc_total_rate_smooth = moving_average(urllc_total_rate, window_size)

axis[0].plot(timesteps[window_size-1:], urllc_total_rate_smooth)
#axis[0,0].plot(timesteps, overall_users_reward)
axis[0].set_xlabel('Timestep')
axis[0].set_ylabel('Data rate (bits/s)')
axis[0].set_title('URLLC Users Total Rate')
axis[0].grid()

axis[1].plot(timesteps[window_size-1:], F_L_inverse_smooth)
#axis[1,0].plot(timesteps, energies)
axis[1].set_title('F_L_inverse_smooth')
axis[1].set_xlabel('Timestep')
axis[1].set_ylabel('Sum energy (J)')
axis[1].grid()

plt.tight_layout()

plt.show()

