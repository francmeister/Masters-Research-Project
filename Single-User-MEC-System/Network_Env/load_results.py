import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
subcarrier_actions = np.load('subcarrier_actions.npy')

#power_actions = np.array(power_actions)
#power_actions = np.squeeze(power_actions)
#print(power_actions)
#print(len(power_actions))

# data to be plotted
episodes = np.arange(1,len(power_actions)+1,1)
 
# plotting
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(episodes, offload_actions, color ="red")
#plt.plot(episodes,subcarrier_actions,color = "blue")
#plt.plot(episodes,offload_actions,color = "green")
plt.legend(["Powers (dBm)","energies","throughputs"])
plt.xlabel("Episodes")
plt.ylabel("Powers (dBm)")
plt.title("Evolution of Power allocations over 500 episodes")
plt.show()

