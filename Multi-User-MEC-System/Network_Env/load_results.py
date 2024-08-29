import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
import math

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
q_action = np.load('q_action.npy')
RBs_actions = np.load('subcarrier_actions.npy')
rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')
rewards_throughput_energy_TD3 = np.load('timestep_rewards_energy_throughput_TD3.npy')
rewards_throughput_energy_DDPG = np.load('timestep_rewards_energy_throughput_DDPG.npy')
#evaluations = np.load('TD3_NetworkEnv-v0_0.npy')
TD3_rewards_throughput_energy = np.load('TD3_NetworkEnv-v0_0.npy')
evaluations_TD3 = np.load('TD3_NetworkEnv-v0_0_TD3.npy')
evaluations_DDPG = np.load('TD3_NetworkEnv-v0_0_DDPG.npy')
rewards_throughput_energy_ = np.load('TD3_NetworkEnv-v0_0_.npy')
allocated_RBs = np.load('allocated_RBs.npy')
fairness_index = np.load('fairnes_index.npy')
sum_allocations_per_RB_matrix = np.load('sum_allocations_per_RB_matrix.npy')
RB_allocation_matrix = np.load('RB_allocation_matrix.npy')
delays = np.load('delays.npy')
tasks_dropped = np.load('tasks_dropped.npy')
outage_probabilities = np.load('outage_probabilties.npy')
urllc_reliability_reward = np.load('urllc_reliability_reward_normalized.npy')
resource_allocation_matrix = np.load('resource_allocation_matrix.npy',allow_pickle=True)
resource_allocation_constraint_violation_count = np.load('resource_allocation_constraint_violation_count.npy',allow_pickle=True)
resource_allocation_matrix = np.array(resource_allocation_matrix)
individual_energies = np.load('individual_energy_rewards.npy')
individual_channel_rates = np.load('individual_channel_rate_rewards.npy')
individual_queue_delays = np.load('individual_queue_delays.npy')
individual_tasks_dropped = np.load('individual_tasks_dropped.npy')
individual_delay_rewards = np.load('individual_delay_rewards.npy')
individual_battery_energy_rewards = np.load('individual_channel_battery_energy_rewards.npy')
individual_total_reward = np.load('individual_total_reward.npy')

individual_local_queue_delays = np.load('individual_local_queue_delays.npy')
individual_offload_queue_delays = np.load('individual_offload_queue_delays.npy')
individual_local_queue_lengths = np.load('individual_local_queue_lengths.npy')
individual_offload_queue_lengths = np.load('individual_offload_queue_lengths.npy')
overall_users_reward = np.load('overall_users_reward.npy')

log_throughput_reward = np.load('throughput_log_reward.npy')






# rewards_throughput_energy_access_point_1 = np.load('timestep_rewards_energy_throughput (3).npy')
# rewards_throughput_energy_access_point_2 = np.load('timestep_rewards_energy_throughput (4).npy')
# rewards_throughput_energy_access_point_3 = np.load('timestep_rewards_energy_throughput (5).npy')
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
delays = rewards_throughput_energy[:,4]

timesteps_TD3 = rewards_throughput_energy_TD3[:,0]
#timesteps_TD3 =TD3_rewards_throughput_energy[:,0]
timesteps_DDPG = rewards_throughput_energy_DDPG[:,0]
#timesteps_DDPG = rewards_throughput_energy_DDPG[:,0]

rewards_TD3 = rewards_throughput_energy_TD3[:,1]
#rewards_TD3 = TD3_rewards_throughput_energy[:,1]
rewards_DDPG = rewards_throughput_energy_DDPG[:,1]
#rewards_DDPG = rewards

timesteps_1_ = rewards_throughput_energy_[:,0]
rewards_1_ = rewards_throughput_energy_[:,1]

# timesteps_1 = rewards_throughput_energy_access_point_1[:,0]
# timesteps_2 = rewards_throughput_energy_access_point_2[:,0]
# timesteps_3 = rewards_throughput_energy_access_point_3[:,0]

# rewards_1 = rewards_throughput_energy_access_point_1[:,1]
# rewards_2 = rewards_throughput_energy_access_point_2[:,1]
# rewards_3 = rewards_throughput_energy_access_point_3[:,1]

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
# evaluation_timesteps = []
# for i in range(0,len(evaluations)):
#     evaluation_timesteps.append(i)
# plt.plot(evaluation_timesteps, evaluations, color ="blue")
# plt.title('Evaluations')
# plt.plot(evaluation_timesteps, evaluations_TD3, color="blue", label='TD3')
# plt.plot(evaluation_timesteps, evaluations_DDPG, color="green", label="DDPG")
# plt.legend()
# plt.title('Individual Total Reward')
#plt.plot(timesteps,energies,color = "blue")
#plt.plot(timesteps,throughputs,color = "green")
#plt.scatter(timesteps,offload_actions,color="blue")
#plt.scatter(timesteps,power_actions,color="green")
#plt.scatter(timesteps,subcarrier_actions,color="red")
figure, axis = plt.subplots(3,3)

# axis[0].plot(timesteps, throughputs)
# axis[0].set_title('throughputs reward')
# axis[0].plot(timesteps, battery_energy_rewards)
# axis[0].set_title('battery energies reward')

# axis[0].plot(timesteps_DDPG, rewards_DDPG)
# axis[0].set_title('DDPG Reward DF = 0.99')

# axis[1].plot(timesteps, rewards)
# axis[1].set_title('DDPG Reward DF = 0.75')

# normalized_rewards_DDPG = []

# for x in rewards_DDPG:
#     normalized_rewards_DDPG.append(interp(x,[0,max(rewards_DDPG)],[0,300]))

# def moving_average(data, window_size):
#     """Compute the moving average of data."""
#     weights = np.repeat(1.0, window_size) / window_size
#     return np.convolve(data, weights, 'valid')

# window_size = 100

# TD3_smooth = moving_average(rewards_TD3, window_size)
# DDPG_smooth = moving_average(normalized_rewards_DDPG, window_size)

# print(len(timesteps_TD3))
# new_timesteps_TD3 = []
# count = 0
# for timestep in timesteps_TD3:
#     new_timesteps_TD3.append(count)
#     count+=1
# #plt.plot(timesteps, rewards, color="green", label="TD3")
# plt.plot(new_timesteps_TD3[window_size-1:], TD3_smooth, color="green", label="TD3")
# plt.plot(new_timesteps_TD3[window_size-1:], DDPG_smooth, color="blue", label='DDPG')
# plt.xlabel("Episodes")
# plt.ylabel("System Reward($\mathcal{R}$)")
# plt.legend(["TD3","DDPG"], loc="upper left")
# plt.grid()

# axis[2].plot(timesteps, RBs_actions)
# axis[2].set_title('RB allocation actions')

# axis[0].plot(timesteps, offload_actions)
# axis[0].set_title('offload actions')

# axis[1].plot(timesteps, power_actions)
# axis[1].set_title('power actions')

# axis[2].plot(timesteps, RBs_actions)
# axis[2].set_title('RB allocation actions')

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


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

rewards_smooth = moving_average(rewards, window_size)
energies_smooth = moving_average(energies, window_size)
throughputs_smooth = moving_average(throughputs, window_size)
delays_smooth = moving_average(delays, window_size)
energy_efficiency_smooth = moving_average(energy_efficiency_rewards, window_size)
fairness_index_smooth = moving_average(fairness_index, window_size)
overall_users_reward_smooth = moving_average(overall_users_reward, window_size)
log_throughput_reward_smooth = moving_average(log_throughput_reward, window_size)
print('len(overall_users_reward): ', len(throughputs))
print('len(throughputs): ', len(overall_users_reward))
print('len(energies): ', len(energies))
print('last reward value: ', overall_users_reward[len(overall_users_reward)-1])
print('last throughput value: ', throughputs[len(throughputs)-1])
print('last energy value: ', energies[len(energies)-1]*10**9)
print('throughput - q_energy*energy: ', throughputs[len(throughputs)-1] - energies[len(energies)-1]*10**8)
# q_action_smooth = moving_average(q_action, window_size)
# offload_actions_smooth = moving_average(offload_actions, window_size)
# power_actions_smooth = moving_average(power_actions, window_size)
# RBs_smooth = moving_average(RBs_actions, window_size)
#print(rewards_smooth)
axis[0,0].plot(timesteps[window_size-1:], overall_users_reward_smooth)
#axis[0,0].plot(timesteps, overall_users_reward)
axis[0,0].set_title('total reward')
axis[0,0].grid()

axis[1,0].plot(timesteps[window_size-1:], energies_smooth)
#axis[1,0].plot(timesteps, energies)
axis[1,0].set_title('Total Energy Consumption')
axis[1,0].set_xlabel('Timestep')
axis[1,0].set_ylabel('Sum energy (J)')
axis[1,0].grid()

axis[0,1].plot(timesteps[window_size-1:], throughputs_smooth)
#axis[0,1].plot(timesteps, throughputs)
axis[0,1].set_title('Total achieved throughput')
axis[0,1].set_xlabel('Timestep')
axis[0,1].set_ylabel('Data Rate (bits/s)')
axis[0,1].grid()

axis[1,1].plot(timesteps[window_size-1:], delays_smooth)
axis[1,1].set_title('Sum delays')
axis[1,1].set_xlabel('Timestep')
axis[1,1].set_ylabel('Delay (ms)')
axis[1,1].grid()

axis[2,0].plot(timesteps[window_size-1:], fairness_index_smooth)
axis[2,0].set_title('Fairness Index')
axis[2,0].set_xlabel('Timestep')
axis[2,0].set_ylabel('FI value')
axis[2,0].grid()

axis[2,1].plot(timesteps[window_size-1:], energy_efficiency_smooth)
axis[2,1].set_title('Energy Efficiency')
axis[2,1].set_xlabel('Timestep')
axis[2,1].set_ylabel('EE Value')
axis[2,1].grid()

axis[0,2].plot(timesteps[window_size-1:], log_throughput_reward_smooth)
axis[0,2].set_title('Log Throughput Constraint')
axis[0,2].set_xlabel('Timestep')
#axis[3,0].set_ylabel('Throughput Constraint Value')
axis[0,2].grid()

# axis[0,0].plot(timesteps, offload_actions)
# axis[0,0].set_title('Offloading Actions')
# axis[0,0].set_xlabel('Timestep')
# axis[0,0].set_ylabel('Offloading Ratio')
# axis[0,0].grid()
# print(power_actions)
# power_actions_new = []
# count = 0
# for p in power_actions:
#     p = (math.pow(10,(p/10)))/1000
#     power_actions_new.append(p)

# print(power_actions)
# axis[1,0].plot(timesteps, power_actions)
# axis[1,0].set_title('Power Allocation Actions')
# axis[1,0].set_xlabel('Timestep')
# axis[1,0].set_ylabel('dbm')
# axis[1,0].grid()

# axis[0,1].plot(timesteps, RBs_actions)
# axis[0,1].set_title('Resource Block Actions')
# axis[0,1].set_xlabel('Timestep')
# axis[0,1].set_ylabel('Number of RBs Allocated')
# axis[0,1].grid()

# timesteps_q_action = []

# for x in range(0,len(q_action)):
#     timesteps_q_action.append(x)
# axis[1,1].plot(timesteps_q_action, q_action)
# axis[1,1].set_title('q_action')
# axis[1,1].set_xlabel('Timestep')
# axis[1,1].set_ylabel('q value')
# axis[1,1].grid()

# print(individual_local_queue_lengths)
# axis[0,0].plot(timesteps, individual_offload_queue_lengths)
# axis[0,0].set_title('Offload Queue Lengths')
# axis[0,0].set_xlabel('Timestep')
# axis[0,0].set_ylabel('Number of tasks')
# axis[0,0].grid()

# axis[1,1].plot(timesteps, individual_local_queue_delays)
# axis[1,1].set_title('Local Queue Delay')
# axis[1,1].set_xlabel('Timestep')
# axis[1,1].set_ylabel('Delay (ms)')
# axis[1,1].grid()

# axis[1,0].plot(timesteps, individual_offload_queue_delays)
# axis[1,0].set_title('Offload Queueing Delay')
# axis[1,0].set_xlabel('Timestep')
# axis[1,0].set_ylabel('Delay (ms)')
# axis[1,0].grid()

# axis[0,1].plot(timesteps, individual_local_queue_lengths)
# axis[0,1].set_title('Local Queue Lengths')
# axis[0,1].set_xlabel('Timestep')
# axis[0,1].set_ylabel('Number of tasks')
# axis[0,1].grid()



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

# def moving_average(data, window_size):
#     """Compute the moving average of data."""
#     weights = np.repeat(1.0, window_size) / window_size
#     return np.convolve(data, weights, 'valid')

# window_size = 1000

# TD3_smooth = moving_average(energy_efficiency_rewards, window_size)

evaluation_timesteps1 = []
for i in range(0,len(energy_efficiency_rewards)):
    evaluation_timesteps1.append(i)


# axis[0].plot(evaluation_timesteps1, energy_efficiency_rewards)
# axis[0].set_title('Energy Efficiency')


# axis[1].plot(evaluation_timesteps1[window_size-1:], TD3_smooth)
# axis[1].set_title('Energy Efficiency')

#plt.plot(timesteps_TD3[window_size-1:], TD3_smooth, color="blue", label='TD3')

plt.tight_layout()

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


