import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
rewards_energy_throughput_0_15 = np.load('timestep_rewards_energy_throughput_0_15.npy')
rewards_energy_throughput_0_35 = np.load('timestep_rewards_energy_throughput_0_35.npy')
rewards_energy_throughput_0_45 = np.load('timestep_rewards_energy_throughput_0_45.npy')
rewards_energy_throughput_0_65 = np.load('timestep_rewards_energy_throughput_0_65.npy')
rewards_energy_throughput_0_99 = np.load('timestep_rewards_energy_throughput_0_99.npy')
rewards_throughput_energy_9_user = np.load('timestep_rewards_energy_throughput_9_Users.npy')
rewards_throughput_energy_11_user = np.load('timestep_rewards_energy_throughput_11_Users.npy')

fairness_index = np.load('fairnes_index.npy')




#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps_0_15 = rewards_energy_throughput_0_15[:,0]
timesteps_0_35 = rewards_energy_throughput_0_35[:,0]
timesteps_0_45 = rewards_energy_throughput_0_45[:,0]
timesteps_0_65 = rewards_energy_throughput_0_65[:,0]
timesteps_0_99 = rewards_energy_throughput_0_99[:,0]
timesteps_9_users = rewards_throughput_energy_9_user[:,0]
timesteps_11_users = rewards_throughput_energy_11_user[:,0]

rewards_0_15 = rewards_energy_throughput_0_15[:,1]
rewards_0_35 = rewards_energy_throughput_0_35[:,1]
rewards_0_45 = rewards_energy_throughput_0_45[:,1]
rewards_0_65 = rewards_energy_throughput_0_65[:,1]
rewards_0_99 = rewards_energy_throughput_0_99[:,1]
rewards_9_users = rewards_throughput_energy_9_user[:,1]
rewards_11_users = rewards_throughput_energy_11_user[:,1]


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

rewards_0_15_smooth = moving_average(rewards_0_15, window_size)
rewards_0_35_smooth = moving_average(rewards_0_35, window_size)
rewards_0_45_smooth = moving_average(rewards_0_45, window_size)
rewards_0_99_smooth = moving_average(rewards_0_99, window_size)
rewards_0_65_smooth = moving_average(rewards_0_65, window_size)
rewards_9_users_smooth = moving_average(rewards_9_users, window_size)
rewards_11_users_smooth = moving_average(rewards_11_users, window_size)


new_timesteps = []
count = 0
for timestep in timesteps_0_15:
    new_timesteps.append(count)
    count+=1

plt.plot(new_timesteps[window_size-1:], rewards_0_15_smooth, color="green", label="0.15")
plt.plot(new_timesteps[window_size-1:], rewards_0_35_smooth, color="brown", label="0.35")
#plt.plot(timesteps_0_45[window_size-1:], rewards_0_45_smooth, color="brown", label='0.45')
plt.plot(new_timesteps[window_size-1:], rewards_0_65_smooth, color="blue", label='0.65')
plt.plot(new_timesteps[window_size-1:], rewards_0_99_smooth, color="red", label='0.99')
#plt.plot(timesteps_9_users[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Episodes")
plt.ylabel("System Reward($\mathcal{R}$)")
plt.legend(["$\epsilon$ = 0.15", "$\epsilon$ = 0.35", "$\epsilon$ = 0.65", "$\epsilon$ = 0.99"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
