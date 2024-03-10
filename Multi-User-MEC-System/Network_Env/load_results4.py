import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
RBs_actions = np.load('subcarrier_actions.npy')
rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')
evaluations = np.load('TD3_NetworkEnv-v0_0.npy')
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

timesteps = rewards_throughput_energy[:,0]
rewards = rewards_throughput_energy[:,1]
energies = rewards_throughput_energy[:,2]
throughputs = rewards_throughput_energy[:,3]


def individual_sub_plots(numbers_users, timesteps, reward_component, string_reward_component):
    row = 0
    col = 0
    if numbers_users == 3:
        row = 2
        col = 2
    elif numbers_users == 5:
        row = 3
        col = 2
    elif numbers_users == 7:
        row = 3
        col = 3
    elif numbers_users == 9:
        row = 3
        col = 3
    elif numbers_users == 11:
        row = 4
        col = 3

    figure, axis = plt.subplots(row,col)
    axis = axis.flatten()
    
    user_num = 0
    for i in range(0, numbers_users):
        user_num = i+1
        axis_title = 'User: ' + str(user_num) + ' ' + string_reward_component
        axis[i].plot(timesteps, reward_component[:, i])
        axis[i].set_title(axis_title)

    plt.tight_layout()
    plt.show()


def individual_user_subplots(user_num, timesteps, total_rewards, energy_rewards, throughput_rewards, battery_energy_rewards, delay_rewards, offload_actions, power_actions, RB_actions):
    row = 3
    col = 3

    figure, axis = plt.subplots(row,col)
    axis = axis.flatten()

    axis[0].plot(timesteps, total_rewards[:,user_num])
    axis[0].set_title('user num: '+ str(user_num) + ' total reward')

    axis[1].plot(timesteps, energy_rewards[:,user_num])
    axis[1].set_title('user num: '+ str(user_num) + ' energy_rewards')

    axis[2].plot(timesteps, throughput_rewards[:,user_num])
    axis[2].set_title('user num: '+ str(user_num) + ' throughput_rewards')

    axis[3].plot(timesteps, battery_energy_rewards[:,user_num])
    axis[3].set_title('user num: '+ str(user_num) + ' battery_energy_rewards')

    axis[4].plot(timesteps, delay_rewards[:,user_num])
    axis[4].set_title('user num: '+ str(user_num) + ' delay_rewards')

    axis[5].plot(timesteps, offload_actions[:,user_num])
    axis[5].set_title('user num: '+ str(user_num) + ' offload_actions')

    axis[6].plot(timesteps, power_actions[:,user_num])
    axis[6].set_title('user num: '+ str(user_num) + ' power actions')

    axis[7].plot(timesteps, RB_actions[:,user_num])
    axis[7].set_title('user num: '+ str(user_num) + ' RB allocation action')

    plt.tight_layout()
    plt.show()


string_reward_component = 'RB allocations'
#individual_sub_plots(numbers_users=len(RBs_actions[0]),timesteps=timesteps,reward_component=RBs_actions,string_reward_component=string_reward_component)

user_num = 1
individual_user_subplots(user_num, timesteps, individual_total_reward, individual_energies, individual_channel_rates, individual_battery_energy_rewards, individual_delay_rewards, offload_actions, power_actions, RBs_actions)