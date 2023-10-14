import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
subcarrier_actions = np.load('subcarrier_actions.npy')
rewards_throughput_energy = np.load('TD3_NetworkEnv-v0_0.npy')

timesteps = rewards_throughput_energy[:,0]
rewards = rewards_throughput_energy[:,1]
energies = rewards_throughput_energy[:,2]
throughputs = rewards_throughput_energy[:,3]

range_start = 800
range_finish = 4500

timesteps_ = timesteps[range_start:range_finish]
rewards_ = rewards[range_start:range_finish]
energies_ = energies[range_start:range_finish]
throughputs_ = throughputs[range_start:range_finish]

offload_actions_ = offload_actions[range_start:range_finish]
power_actions_ = power_actions[range_start:range_finish]
subcarrier_actions_ = subcarrier_actions[range_start:range_finish]

print(len(timesteps_))

figure, axis = plt.subplots(6,1)
#plt.plot(timesteps, rewards,color = "blue")


axis[0].plot(timesteps, energies)
axis[0].set_title('energies reward')

axis[1].plot(timesteps, throughputs)
axis[1].set_title('throughputs reward')

axis[2].plot(timesteps, rewards)
axis[2].set_title('total reward')

axis[3].scatter(timesteps, offload_actions)
axis[3].set_title('offlaoding actions')

axis[4].scatter(timesteps, power_actions)
axis[4].set_title('power actions')

axis[5].scatter(timesteps, subcarrier_actions)
axis[5].set_title('subcarrier actions')

plt.show()