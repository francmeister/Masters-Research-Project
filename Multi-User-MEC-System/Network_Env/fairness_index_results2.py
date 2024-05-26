import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


timestep_rewards_energy_throughput_1 = np.load('timestep_rewards_energy_throughput_1.npy')
timestep_rewards_energy_throughput_3 = np.load('timestep_rewards_energy_throughput_3.npy')
timestep_rewards_energy_throughput_5 = np.load('timestep_rewards_energy_throughput_5.npy')
timestep_rewards_energy_throughput_7 = np.load('timestep_rewards_energy_throughput_7.npy')
timestep_rewards_energy_throughput_13 = np.load('timestep_rewards_energy_throughput_13.npy')


fairnes_index_1 = np.load('fairnes_index_1.npy') 
fairnes_index_3 = np.load('fairnes_index_3.npy') 
fairnes_index_5 = np.load('fairnes_index_5.npy') 
fairnes_index_7 = np.load('fairnes_index_7.npy') 
fairnes_index_13 = np.load('fairnes_index_13.npy') 

timesteps_1 = timestep_rewards_energy_throughput_1[:,0]
timesteps_3 = timestep_rewards_energy_throughput_3[:,0]
timesteps_5 = timestep_rewards_energy_throughput_5[:,0]
timesteps_7 = timestep_rewards_energy_throughput_7[:,0]
timesteps_13 = timestep_rewards_energy_throughput_13[:,0]

rewards_1 = timestep_rewards_energy_throughput_1[:,1]
rewards_3 = timestep_rewards_energy_throughput_3[:,1]
rewards_5 = timestep_rewards_energy_throughput_5[:,1]
rewards_7 = timestep_rewards_energy_throughput_7[:,1]
rewards_13 = timestep_rewards_energy_throughput_13[:,1]

throughput_1 = timestep_rewards_energy_throughput_1[:,3]
throughput_3 = timestep_rewards_energy_throughput_3[:,3]
throughput_5 = timestep_rewards_energy_throughput_5[:,3]
throughput_7 = timestep_rewards_energy_throughput_7[:,3]
throughput_13 = timestep_rewards_energy_throughput_13[:,3]

energy_1 = timestep_rewards_energy_throughput_1[:,2]
energy_3 = timestep_rewards_energy_throughput_3[:,2]
energy_5 = timestep_rewards_energy_throughput_5[:,2]
energy_7 = timestep_rewards_energy_throughput_7[:,2]
energy_13 = timestep_rewards_energy_throughput_13[:,2]




def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 1000

rewards_1_smooth = moving_average(rewards_1, window_size)
rewards_3_smooth = moving_average(rewards_3, window_size)
rewards_5_smooth = moving_average(rewards_5, window_size)
rewards_7_smooth = moving_average(rewards_7, window_size)
rewards_13_smooth = moving_average(rewards_13, window_size)

throughput_1_smooth = moving_average(throughput_1, window_size)
throughput_3_smooth = moving_average(throughput_3, window_size)
throughput_5_smooth = moving_average(throughput_5, window_size)
throughput_7_smooth = moving_average(throughput_7, window_size)
throughput_13_smooth = moving_average(throughput_13, window_size)

energy_1_smooth = moving_average(energy_1, window_size)
energy_3_smooth = moving_average(energy_3, window_size)
energy_5_smooth = moving_average(energy_5, window_size)
energy_7_smooth = moving_average(energy_7, window_size)
energy_13_smooth = moving_average(energy_13, window_size)

fairness_index_1_smooth = moving_average(fairnes_index_1, window_size)
fairness_index_3_smooth = moving_average(fairnes_index_3, window_size)
fairness_index_5_smooth = moving_average(fairnes_index_5, window_size)
fairness_index_7_smooth = moving_average(fairnes_index_7, window_size)
fairness_index_13_smooth = moving_average(fairnes_index_13, window_size)


# plt.plot(timesteps_0_1[window_size-1:], rewards_0_1_smooth, color="green", label="1 User")
# plt.plot(timesteps_0_2[window_size-1:], rewards_0_2_smooth, color="blue", label='3 Users')
# plt.plot(timesteps_0_4[window_size-1:], rewards_0_4_smooth, color="red", label='5 Users')
# plt.plot(timesteps_0_8[window_size-1:], rewards_0_8_smooth, color="purple", label='7 Users')
# plt.plot(timesteps_1[window_size-1:], rewards_1_smooth, color="grey", label='9 Users')

figure, axis = plt.subplots(2,2)

axis[0,0].plot(timesteps_1[window_size-1:], rewards_1_smooth, color="green")
axis[0,0].plot(timesteps_3[window_size-1:], rewards_3_smooth, color="blue")
axis[0,0].plot(timesteps_5[window_size-1:], rewards_5_smooth, color="red")
axis[0,0].plot(timesteps_7[window_size-1:], rewards_7_smooth, color="purple")
axis[0,0].plot(timesteps_13[window_size-1:], rewards_13_smooth, color="grey")
axis[0,0].legend(["FI=1","FI=3", "FI=5", "FI=7", "FI=13"], loc="upper left")
axis[0,0].set_title('Total Reward')
axis[0,0].grid()

axis[0,1].plot(timesteps_1[window_size-1:], fairness_index_1_smooth, color="green")
axis[0,1].plot(timesteps_3[window_size-1:], fairness_index_3_smooth, color="blue")
axis[0,1].plot(timesteps_5[window_size-1:], fairness_index_5_smooth, color="red")
axis[0,1].plot(timesteps_7[window_size-1:], fairness_index_7_smooth, color="purple")
axis[0,1].plot(timesteps_13[window_size-1:], fairness_index_13_smooth, color="grey")
axis[0,1].legend(["FI=1","FI=3", "FI=5", "FI=7", "FI=13"], loc="upper left")
axis[0,1].set_title('Fairness Index')
axis[0,1].grid()

axis[1,1].plot(timesteps_1[window_size-1:], throughput_1_smooth, color="green")
axis[1,1].plot(timesteps_3[window_size-1:], throughput_3_smooth, color="blue")
axis[1,1].plot(timesteps_5[window_size-1:], throughput_5_smooth, color="red")
axis[1,1].plot(timesteps_7[window_size-1:], throughput_7_smooth, color="purple")
axis[1,1].plot(timesteps_13[window_size-1:], throughput_13_smooth, color="grey")
axis[1,1].legend(["FI=1","FI=3", "FI=5", "FI=7", "FI=13"], loc="upper left")
axis[1,1].set_title('Throughput')
axis[1,1].grid()

axis[1,0].plot(timesteps_1[window_size-1:], energy_1_smooth, color="green")
axis[1,0].plot(timesteps_3[window_size-1:], energy_3_smooth, color="blue")
axis[1,0].plot(timesteps_5[window_size-1:], energy_5_smooth, color="red")
axis[1,0].plot(timesteps_7[window_size-1:], energy_7_smooth, color="purple")
axis[1,0].plot(timesteps_13[window_size-1:], energy_13_smooth, color="grey")
axis[1,0].legend(["FI=1","FI=3", "FI=5", "FI=7", "FI=13"], loc="upper left")
axis[1,0].set_title('Energy')
axis[1,0].grid()

plt.xlabel("Timestep(t)")
#plt.ylabel("Reward")
#plt.legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")
#plt.grid()

plt.tight_layout()

plt.show()
