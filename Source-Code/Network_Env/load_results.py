import numpy as np
import matplotlib.pyplot as plt

a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
subcarrier_actions = np.load('subcarrier_actions.npy')

print(subcarrier_actions)
print('fdfdf')

# data to be plotted
timesteps = a_load[:,0]
rewards = a_load[:,1]
energies = a_load[:,2]
throughputs = a_load[:,3]
 
# plotting
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(timesteps, rewards, color ="red")
plt.plot(timesteps,energies,color = "blue")
plt.plot(timesteps,throughputs,color = "green")
plt.legend(["rewards","energies","throughputs"])
plt.xlabel("Time Steps")
plt.ylabel("Normalized Reward, Energy Consumption, Throughput")
plt.title("Graph showing the evolution of reward, energy consumption and throughput over the timesteps")
plt.show()
