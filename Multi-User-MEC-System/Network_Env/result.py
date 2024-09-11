import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

result = np.load('sum_allocations_per_RB_matrix.npy')
rewards_throughput_energy_DDPG = np.load('timestep_rewards_energy_throughput_DDPG.npy')
rewards_throughput_energy_TD3 = np.load('timestep_rewards_energy_throughput_TD3.npy')

timesteps_TD3 = rewards_throughput_energy_TD3[:,0]
#timesteps_TD3 =TD3_rewards_throughput_energy[:,0]
timesteps_DDPG = rewards_throughput_energy_DDPG[:,0]
#timesteps_DDPG = rewards_throughput_energy_DDPG[:,0]

rewards_TD3 = rewards_throughput_energy_TD3[:,1]
#rewards_TD3 = TD3_rewards_throughput_energy[:,1]
rewards_DDPG = rewards_throughput_energy_DDPG[:,1]


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

normalized_rewards_DDPG = []
normalized_rewards_TD3 = []

for x in rewards_DDPG:
    normalized_rewards_DDPG.append(interp(x,[0,max(rewards_DDPG)],[0,0.6]))

for x in rewards_TD3:
    normalized_rewards_TD3.append(interp(x,[0,max(rewards_TD3)],[0,1]))

TD3_smooth = moving_average(normalized_rewards_TD3, window_size)
DDPG_smooth = moving_average(normalized_rewards_DDPG, window_size)

new_timesteps_TD3 = []
count = 0
for timestep in timesteps_TD3:
    new_timesteps_TD3.append(count)
    count+=1

new_timesteps_DDPG = []
count = 0
for timestep in timesteps_DDPG:
    new_timesteps_DDPG.append(count)
    count+=1

plt.plot(new_timesteps_TD3[window_size-1:], TD3_smooth, color="green")
plt.plot(new_timesteps_DDPG[window_size-1:], DDPG_smooth, color="blue")
plt.xlabel("Episodes")
plt.ylabel("Normalized System Reward")
plt.legend(["TD3","DDPG"], loc="upper left")
plt.grid()

plt.show()

#print(timesteps)

print(len(result))
#print(len(timesteps))

print(result)