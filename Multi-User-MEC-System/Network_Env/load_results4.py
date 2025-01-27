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

individual_energy_harvested = np.load('individual_energy_harvested.npy')
individual_battery_energy_levels = np.load('individual_battery_energy_levels.npy')
individual_average_offloading_rates = np.load('individual_average_offloading_rates.npy')
print('individual_average_offloading_rates:')
print(individual_average_offloading_rates[0])

individual_large_scale_gains = np.load('individual_large_scale_gains.npy')
individual_small_scale_gains = np.load('individual_small_scale_gains.npy')
individual_local_queue_length_num_tasks = np.load('individual_local_queue_length_num_tasks.npy')
individual_offload_queue_length_num_tasks = np.load('individual_offload_queue_length_num_tasks.npy')

individual_simulation_local_queueing_delay = np.load('individual_simulation_local_queueing_delay.npy')
individual_simulation_offload_queueing_delay = np.load('individual_simulation_offload_queueing_delay.npy')
individual_simulation_total_delay = np.load('individual_simulation_total_delay.npy')




#print(individual_battery_energy_levels)

#print(individual_expected_rate_over_prev_T_slot[:,0])

# 2D Matrices below. individual_energies has energy results for each user. Each column represents a user
# Each row represents energy values for all users in a time slot
# Same goes for the other individual matrices

# print(individual_energies)
# [[1.99847734 2.73725381 0.83975129 ... 1.6791527  0.40350153 2.32592202]
#  [1.66163198 1.18464085 1.06151078 ... 5.4206672  0.43677021 2.25944429]
#  [1.87265394 0.45064082 5.04641513 ... 1.29379751 0.4562806  5.12412495]
#  ...
#  [1.85947439 0.2297878  0.85704297 ... 1.45300654 0.24951508 1.18069556]
#  [1.55156814 0.4136556  0.98747191 ... 1.82424129 0.42132404 1.25065084]
#  [1.94428997 0.05607016 0.75602795 ... 1.92813701 0.         1.21327016]]

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
individual_average_task_size_offload_queue = np.load('individual_average_task_size_offload_queue.npy')
individual_expected_rate_over_prev_T_slot = np.load('individual_expected_rate_over_prev_T_slot.npy')
individual_local_energy_consumed = np.load('individual_local_energy_consumed.npy')
individual_offloading_energy = np.load('individual_offloading_energy.npy')

individual_embb_puncturing_users_sum_data_rates = np.load('individual_embb_puncturing_users_sum_data_rates.npy')
individual_embb_num_puncturing_users = np.load('individual_embb_num_puncturing_users.npy')
#print(individual_offload_queue_lengths[0,:])

users_lc_service_rates = np.load('users_lc_service_rates.npy')
#print(users_lc_service_rates)

timesteps = rewards_throughput_energy[:,0]
rewards = rewards_throughput_energy[:,1]
energies = rewards_throughput_energy[:,2]
throughputs = rewards_throughput_energy[:,3]

user_1_individual_average_task_size_offload_queue = individual_average_task_size_offload_queue[:,0]
user_1_individual_expected_rate_over_prev_T_slot = individual_expected_rate_over_prev_T_slot[:,0]

user_1_offload_delay = individual_offload_queue_delays[:,0]
user_1_throughput = individual_channel_rates[:,0]
user_1_offload_queue_length = individual_offload_queue_lengths[:,0]

print('len(user_1_offload_delay): ', len(user_1_offload_delay))
print('len(user_1_throughput): ',len(user_1_throughput))
print('len(user_1_offload_queue_length): ',len(user_1_offload_queue_length))
print('len(user_1_individual_expected_rate_over_prev_T_slot): ',len(user_1_individual_expected_rate_over_prev_T_slot))
print('len(user_1_individual_average_task_size_offload_queue): ',len(user_1_individual_average_task_size_offload_queue))
index_of_max = np.argmax(user_1_offload_delay)
print('max delay: ', user_1_offload_delay[index_of_max])
print('offload queue length at max delay: ', user_1_offload_queue_length[index_of_max])
print("user's average throughput where delay is max: ", user_1_individual_expected_rate_over_prev_T_slot[index_of_max])
print("user's average task size on offload queue where delay is max: ", user_1_individual_average_task_size_offload_queue[index_of_max])

def individual_sub_plots(numbers_users, timesteps, reward_component, string_reward_component):
    row = 0
    col = 0
    if numbers_users == 1 or numbers_users == 2 or numbers_users == 3 or numbers_users == 4:
        row = 2
        col = 2
    elif numbers_users == 5 or numbers_users == 6:
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
    
    print('len(timesteps): ', len(timesteps))
    x_point = timesteps[15]
    if x_point in timesteps:
        index = np.where(timesteps == x_point)[0][0]

        y_values = reward_component[index, :]
        print(f"Y-values at x = {x_point}: {y_values}")

        user_num = 0
        for i in range(0, numbers_users):
            user_num = i+1
            axis_title = 'User: ' + str(user_num) + ' ' + string_reward_component
            axis[i].plot(timesteps, reward_component[:, i], linestyle='-', marker='o')
            axis[i].set_title(axis_title)

            # Highlight the point on the plot
            axis[i].plot(x_point, reward_component[index, i], 'ro')  # Red dot for the specified x-point
            axis[i].annotate(f"{reward_component[index, i]:.2f}", 
                            (x_point, reward_component[index, i]), 
                            textcoords="offset points", 
                            xytext=(5,5), 
                            ha='center')

    plt.tight_layout()
    plt.show()

def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def individual_user_subplots(user_num, timesteps, energy_rewards, throughput_rewards, delay_rewards, offload_actions, power_actions, local_queue_length, local_queue_delay,offload_queue_length,offload_queue_delay, RBs_actions,individual_average_task_size_offload_queue,individual_expected_rate_over_prev_T_slot,individual_battery_energy_levels,
                             individual_energy_harvested,individual_local_queue_length_num_tasks,individual_offload_queue_length_num_tasks,
                             individual_local_energy_consumed,individual_offloading_energy,individual_simulation_total_delay,individual_simulation_offload_queueing_delay,individual_simulation_local_queueing_delay):
    row = 4
    col = 5

    figure, axis = plt.subplots(row,col)
    axis = axis.flatten()

    # axis[0].plot(timesteps, total_rewards[:,user_num])
    # axis[0].set_title('user num: '+ str(user_num) + ' total reward')
    window_size = 5
    energy_rewards_smooth = moving_average(energy_rewards[:,user_num], window_size)

    axis[0].plot(timesteps[window_size-1:], energy_rewards_smooth)
    axis[0].set_title('energy consumption(J)')

    axis[1].plot(timesteps, throughput_rewards[:,user_num])
    axis[1].set_title('data rate (bits/s)')

    # axis[3].plot(timesteps, battery_energy_rewards[:,user_num])
    # axis[3].set_title('user num: '+ str(user_num) + ' battery_energy_rewards')

    axis[2].plot(timesteps, delay_rewards[:,user_num])
    axis[2].set_title('delay(ms)')

    axis[3].plot(timesteps, offload_actions[:,user_num])
    axis[3].set_title('offload_actions')

    axis[4].plot(timesteps, power_actions[:,user_num])
    axis[4].set_title('power actions')

    # axis[5].plot(timesteps, power_actions[:,user_num])
    # axis[5].set_title('user num: '+ str(user_num) + ' power actions')

    # axis[6].plot(timesteps, RBs_actions[:,user_num])
    # axis[6].set_title('user num: '+ str(user_num) + ' RB allocation action')

    axis[5].plot(timesteps, local_queue_length[:,user_num])
    axis[5].set_title('local_queue_length')

    axis[6].plot(timesteps, local_queue_delay[:,user_num])
    axis[6].set_title('local_queue_delay(ms)')

    axis[7].plot(timesteps, offload_queue_length[:,user_num])
    axis[7].set_title('offload_queue_length')

    axis[8].plot(timesteps, offload_queue_delay[:,user_num])
    axis[8].set_title('offload_queue_delay(ms)')

    axis[9].plot(timesteps, individual_battery_energy_levels[:,user_num])
    axis[9].set_title('Battery Energy Level(J)')

    axis[10].plot(timesteps, individual_energy_harvested[:,user_num])
    axis[10].set_title('Energy Harvested(J)')

    axis[11].plot(timesteps, RBs_actions[:,user_num])
    axis[11].set_title('RBs')

    axis[12].plot(timesteps, individual_local_queue_length_num_tasks[:,user_num])
    axis[12].set_title('local_queue_length_tasks')

    axis[13].plot(timesteps, individual_offload_queue_length_num_tasks[:,user_num])
    axis[13].set_title('offload_queue_length_tasks')

    individual_local_energy_consumed_smooth = moving_average(individual_local_energy_consumed[:,user_num], window_size)

    axis[14].plot(timesteps[window_size-1:], individual_local_energy_consumed_smooth)
    axis[14].set_title('local_energy')

    axis[15].plot(timesteps, individual_offloading_energy[:,user_num])
    axis[15].set_title('offloading_energy')

    axis[16].plot(timesteps, individual_simulation_total_delay[:,user_num])
    axis[16].set_title('simulation delay')

    axis[17].plot(timesteps, individual_simulation_offload_queueing_delay[:,user_num])
    axis[17].set_title('simulation offloading delay')

    axis[18].plot(timesteps, individual_simulation_local_queueing_delay[:,user_num])
    axis[18].set_title('simulation local delay')


    plt.tight_layout()
    plt.show()


# offload_actions = np.load('offloading_actions.npy')
# power_actions = np.load('power_actions.npy')
# RBs_actions = np.load('subcarrier_actions.npy')
string_reward_component = 'offload_actions'
print('len(power_actions[0]): ',len(power_actions[0]))
individual_sub_plots(numbers_users=len(power_actions[0]),timesteps=timesteps,reward_component=offload_actions,string_reward_component=string_reward_component)

user_num =1

#individual_user_subplots(user_num, timesteps, individual_energies, individual_channel_rates, individual_queue_delays, offload_actions, power_actions,individual_local_queue_lengths, individual_local_queue_delays, individual_offload_queue_lengths, individual_offload_queue_delays, RBs_actions,individual_average_task_size_offload_queue,individual_expected_rate_over_prev_T_slot,individual_battery_energy_levels,individual_energy_harvested,individual_local_queue_length_num_tasks,individual_offload_queue_length_num_tasks,individual_local_energy_consumed,individual_offloading_energy,individual_simulation_total_delay,individual_simulation_offload_queueing_delay,individual_simulation_local_queueing_delay)

