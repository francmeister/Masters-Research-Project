import numpy as np
import matplotlib.pyplot as plt


individual_battery_energy_levels = np.load('individual_battery_energy_levels.npy')
individual_large_scale_gains = np.load('individual_large_scale_gains.npy')
individual_small_scale_gains = np.load('individual_small_scale_gains.npy')
individual_local_queue_lengths = np.load('individual_local_queue_lengths.npy')
individual_offload_queue_lengths = np.load('individual_offload_queue_lengths.npy')

rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')
timesteps = rewards_throughput_energy[:,0]


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
    
    print('len(timesteps): ', len(timesteps))
    x_point = timesteps[1500]
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


# def individual_user_subplots(user_num, timesteps, energy_rewards, throughput_rewards, delay_rewards, offload_actions, power_actions, local_queue_length, local_queue_delay,offload_queue_length,offload_queue_delay, RBs_actions,individual_average_task_size_offload_queue,individual_expected_rate_over_prev_T_slot,individual_battery_energy_levels,individual_energy_harvested):
#     row = 4
#     col = 3

#     figure, axis = plt.subplots(row,col)
#     axis = axis.flatten()

#     # axis[0].plot(timesteps, total_rewards[:,user_num])
#     # axis[0].set_title('user num: '+ str(user_num) + ' total reward')

#     axis[0].plot(timesteps, energy_rewards[:,user_num])
#     axis[0].set_title('user num: '+ str(user_num) + ' energy consumption (J)')

#     axis[1].plot(timesteps, throughput_rewards[:,user_num])
#     axis[1].set_title('user num: '+ str(user_num) + ' achieved throughput (bits/s)')

#     # axis[3].plot(timesteps, battery_energy_rewards[:,user_num])
#     # axis[3].set_title('user num: '+ str(user_num) + ' battery_energy_rewards')

#     axis[2].plot(timesteps, delay_rewards[:,user_num])
#     axis[2].set_title('user num: '+ str(user_num) + ' delay (ms)')

#     axis[3].plot(timesteps, offload_actions[:,user_num])
#     axis[3].set_title('user num: '+ str(user_num) + ' offload_actions')

#     axis[4].plot(timesteps, power_actions[:,user_num])
#     axis[4].set_title('user num: '+ str(user_num) + ' power actions')

#     # axis[5].plot(timesteps, power_actions[:,user_num])
#     # axis[5].set_title('user num: '+ str(user_num) + ' power actions')

#     # axis[6].plot(timesteps, RBs_actions[:,user_num])
#     # axis[6].set_title('user num: '+ str(user_num) + ' RB allocation action')

#     axis[5].plot(timesteps, local_queue_length[:,user_num])
#     axis[5].set_title('user num: '+ str(user_num) + ' local_queue_length')

#     axis[6].plot(timesteps, local_queue_delay[:,user_num])
#     axis[6].set_title('user num: '+ str(user_num) + ' local_queue_delay (ms)')

#     axis[7].plot(timesteps, offload_queue_length[:,user_num])
#     axis[7].set_title('user num: '+ str(user_num) + ' offload_queue_length')

#     axis[8].plot(timesteps, offload_queue_delay[:,user_num])
#     axis[8].set_title('user num: '+ str(user_num) + ' offload_queue_delay (ms)')

#     axis[9].plot(timesteps, individual_battery_energy_levels[:,user_num])
#     axis[9].set_title('user num: '+ str(user_num) + ' Battery Energy Level (J)')

#     axis[10].plot(timesteps, individual_energy_harvested[:,user_num])
#     axis[10].set_title('user num: '+ str(user_num) + ' Energy Harvested (J)')

#     axis[11].plot(timesteps, RBs_actions[:,user_num])
#     axis[11].set_title('user num: '+ str(user_num) + ' RB allocation action')

#     plt.tight_layout()
#     plt.show()


string_reward_component = 'individual_battery_energy_levels'
print(timesteps)
individual_sub_plots(numbers_users=len(individual_battery_energy_levels[0]),timesteps=timesteps,reward_component=individual_battery_energy_levels,string_reward_component=string_reward_component)

user_num =3
#individual_user_subplots(user_num, timesteps, individual_energies, individual_channel_rates, individual_queue_delays, offload_actions, power_actions,individual_local_queue_lengths, individual_local_queue_delays, individual_offload_queue_lengths, individual_offload_queue_delays, RBs_actions,individual_average_task_size_offload_queue,individual_expected_rate_over_prev_T_slot,individual_battery_energy_levels,individual_energy_harvested)

