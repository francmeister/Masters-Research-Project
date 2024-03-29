import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
#offload_actions = np.load('offloading_actions.npy')
#power_actions = np.load('power_actions.npy')
subcarrier_actions = np.load('subcarrier_actions.npy')
rewards_throughput_energy = np.load('TD3_NetworkEnv-v0_0.npy')
allocated_RBs = np.load('allocated_RBs.npy')
fairness_index = np.load('fairnes_index.npy')
print('allocated RBs')
print(allocated_RBs)
print('fairness index')
print(fairness_index)
#power_actions = np.array(power_actions)
#power_actions = np.squeeze(power_actions)
#print(power_actions)
#print(len(power_actions))
#print('rewards_throughput_energy: ', rewards_throughput_energy)
timesteps = rewards_throughput_energy[:,0]
rewards = rewards_throughput_energy[:,1]
energies = rewards_throughput_energy[:,2]
throughputs = rewards_throughput_energy[:,3]
#print(timesteps)
# data to be plotted
#episodes = np.arange(1,len(power_actions)+1,1)
#print(rewards_throughput_energy)
#print(timesteps)
#print(subcarrier_actions) 
# plotting
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(timesteps, rewards, color ="red")
plt.plot(timesteps,energies,color = "blue")
plt.plot(timesteps,throughputs,color = "green")
#plt.scatter(timesteps,offload_actions,color="blue")
#plt.scatter(timesteps,power_actions,color="green")
#plt.scatter(timesteps,subcarrier_actions,color="red")
plt.legend(["reward", "energy", "throughput"])
plt.xlabel("Episodes")
plt.ylabel("Normalized data")
plt.title("Evolution of Reward")
#plt.show()

