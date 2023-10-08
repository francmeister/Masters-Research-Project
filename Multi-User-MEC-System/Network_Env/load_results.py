import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
subcarrier_actions = np.load('subcarrier_actions.npy')
rewards_throughput_energy = np.load('TD3_NetworkEnv-v0_0.npy')
allocated_RBs = np.load('allocated_RBs.npy')
fairness_index = np.load('fairnes_index.npy')

energy_efficiency_rewards = np.load('energy_efficiency_rewards.npy')
battery_energy_rewards = np.load('energy_rewards.npy')
throughput_rewards = np.load('throughput_rewards.npy')
delay_rewards = np.load('delay_rewards.npy')
print(energy_efficiency_rewards)

#print(power_actions)
#print(fairness_index)
#print(subcarrier_actions)
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
#plt.title("Line graph")
#plt.xlabel("X axis")
#plt.ylabel("Y axis")
#plt.plot(timesteps, fairness_index, color ="red")
#plt.plot(timesteps,energies,color = "blue")
#plt.plot(timesteps,throughputs,color = "green")
#plt.scatter(timesteps,offload_actions,color="blue")
#plt.scatter(timesteps,power_actions,color="green")
#plt.scatter(timesteps,subcarrier_actions,color="red")
figure, axis = plt.subplots(7,1)
'''
axis[0].plot(timesteps, energies)
axis[0].set_title('energies reward')

axis[1].plot(timesteps, throughputs)
axis[1].set_title('throughputs reward')

axis[2].plot(timesteps, rewards)
axis[2].set_title('total reward')
'''


axis[0].plot(timesteps, energies)
axis[0].set_title('energies reward')

axis[1].plot(timesteps, throughput_rewards)
axis[1].set_title('throughputs reward')

axis[2].plot(timesteps, energy_efficiency_rewards)
axis[2].set_title('energy efficiency reward')

axis[3].plot(timesteps, delay_rewards)
axis[3].set_title('delay reward')

axis[4].plot(timesteps, battery_energy_rewards)
axis[4].set_title('battery energy reward')

axis[5].plot(timesteps, rewards)
axis[5].set_title('total reward')

axis[6].plot(timesteps, fairness_index)
axis[6].set_title('fairness index')


plt.show()
#plt.figure(1)
###plt.subplot(211)
#plt.plot(timesteps, fairness_index, color ="red")
#plt.subplot(212)
#plt.plot(timesteps, throughputs, color ="green")
#plt.subplot(214)
#plt.plot(timesteps, rewards, color ="blue")

#plt.legend(["reward"])
#plt.xlabel("Episodes")
#plt.ylabel("reward")
#plt.title("Evolution of Reward")
#plt.show()

