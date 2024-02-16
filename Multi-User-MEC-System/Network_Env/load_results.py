import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
RBs_actions = np.load('subcarrier_actions.npy')
rewards_throughput_energy = np.load('TD3_NetworkEnv-v0_0.npy')
allocated_RBs = np.load('allocated_RBs.npy')
fairness_index = np.load('fairnes_index.npy')
sum_allocations_per_RB_matrix = np.load('sum_allocations_per_RB_matrix.npy')
RB_allocation_matrix = np.load('RB_allocation_matrix.npy')
delays = np.load('delays.npy')
tasks_dropped = np.load('tasks_dropped.npy')
resource_allocation_matrix = np.load('resource_allocation_matrix.npy',allow_pickle=True)
resource_allocation_constraint_violation_count = np.load('resource_allocation_constraint_violation_count.npy',allow_pickle=True)
resource_allocation_matrix = np.array(resource_allocation_matrix)

rewards_throughput_energy_access_point_1 = np.load('timestep_rewards_energy_throughput (3).npy')
rewards_throughput_energy_access_point_2 = np.load('timestep_rewards_energy_throughput (4).npy')
rewards_throughput_energy_access_point_3 = np.load('timestep_rewards_energy_throughput (5).npy')
#print(resource_allocation_matrix[323])
#print(len(resource_allocation_matrix))
#print(RB_allocation_matrix)
#print(RBs_actions)

energy_efficiency_rewards = np.load('energy_efficiency_rewards.npy')
battery_energy_rewards = np.load('battery_energy_rewards.npy')
throughput_rewards = np.load('throughput_rewards.npy')
delay_rewards = np.load('delay_rewards.npy')
#print(energy_efficiency_rewards)
#print(allocated_RBs)
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

timesteps_1 = rewards_throughput_energy_access_point_1[:,0]
timesteps_2 = rewards_throughput_energy_access_point_2[:,0]
timesteps_3 = rewards_throughput_energy_access_point_3[:,0]

rewards_1 = rewards_throughput_energy_access_point_1[:,1]
rewards_2 = rewards_throughput_energy_access_point_2[:,1]
rewards_3 = rewards_throughput_energy_access_point_3[:,1]

start_index = 6200
end_index = 25000

timesteps_ = timesteps[start_index:end_index]
rewards_ = rewards[start_index:end_index]
energies_ = energies[start_index:end_index]
throughputs_ = throughputs[start_index:end_index]
delays_ = delays[start_index:end_index]
tasks_dropped_ = tasks_dropped[start_index:end_index]
delay_rewards_ = delay_rewards[start_index:end_index]
battery_energy_rewards_ = battery_energy_rewards[start_index:end_index]

offload_actions_ = offload_actions[start_index:end_index]
power_actions_ = power_actions[start_index:end_index]
RBs_actions_ = RBs_actions[start_index:end_index]
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
#plt.plot(timesteps, rewards, color ="black")
plt.plot(timesteps, rewards, color ="black")
#plt.plot(timesteps,energies,color = "blue")
#plt.plot(timesteps,throughputs,color = "green")
#plt.scatter(timesteps,offload_actions,color="blue")
#plt.scatter(timesteps,power_actions,color="green")
#plt.scatter(timesteps,subcarrier_actions,color="red")
#figure, axis = plt.subplots(3,1)

# axis[0].plot(timesteps, throughputs)
# axis[0].set_title('throughputs reward')
# axis[0].plot(timesteps, battery_energy_rewards)
# axis[0].set_title('battery energies reward')

# axis[0].plot(timesteps_1, rewards_1)
# axis[0].set_title('Access Point 1')

# axis[1].plot(timesteps_2, rewards_2)
# axis[1].set_title('Access Point 2')

# axis[2].plot(timesteps_3, rewards_3)
# axis[2].set_title('Access Point 3')

# axis[1].plot(timesteps, delay_rewards)
# axis[1].set_title('delay')

# axis[2].plot(timesteps, battery_energy_rewards)
# axis[2].set_title('battery energy')

# axis[2].plot(timesteps, delay_rewards)
# axis[2].set_title('delay reward')

# axis[3].plot(timesteps, battery_energy_rewards)
# axis[3].set_title('battery energy reward')

# axis[3].plot(timesteps, battery_energy_rewards)
# axis[3].set_title('battery_energy_rewards')

# axis[2].plot(timesteps, tasks_dropped)
# axis[2].set_title('tasks dropped')

# axis[2].plot(timesteps, delay_rewards)
# axis[2].set_title('delay')

# axis[1].plot(timesteps, delay_rewards)
# axis[1].set_title('delay rewards')

# axis[2].plot(timesteps, energy_efficiency_rewards)
# axis[2].set_title('Energy efficiency')

# axis[1].scatter(timesteps, offload_actions)
# axis[1].set_title('offloading actions')

# axis[2].scatter(timesteps, power_actions)
# axis[2].set_title('power actions')

# axis[3].scatter(timesteps, RBs_actions)
# axis[3].set_title('power allocation actions')

# axis[4].scatter(timesteps, RBs_actions)
# axis[4].set_title('RB allocation actions')



# axis[0].plot(timesteps, energies)
# axis[0].set_title('energies reward')

# axis[1].plot(timesteps, throughputs)
# axis[1].set_title('throughputs reward')

# axis[2].plot(timesteps, rewards)
# axis[2].set_title('total reward')

# axis[3].plot(timesteps, fairness_index)
# axis[3].set_title('fairness index')

'''
axis[3].scatter(timesteps_, offload_actions_)
axis[3].set_title('offload actions')

axis[4].scatter(timesteps_, power_actions_)
axis[4].set_title('power allocation actions')

axis[5].scatter(timesteps_, RBs_actions_)
axis[5].set_title('RB allocation actions')
'''









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


