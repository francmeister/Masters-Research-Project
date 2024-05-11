import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')


tasks_dropped = np.load('tasks_dropped.npy')
outage_probabilities = np.load('outage_probabilties.npy')
energy_efficiency_rewards = np.load('energy_efficiency_rewards.npy')
battery_energy_rewards = np.load('battery_energy_rewards.npy')
throughput_rewards = np.load('throughput_rewards.npy')
delay_rewards = np.load('delay_rewards.npy')
timesteps_rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')

timesteps = timesteps_rewards_throughput_energy[:,0]
energies = timesteps_rewards_throughput_energy[:,2]
throughputs = timesteps_rewards_throughput_energy[:,3]
rewards = timesteps_rewards_throughput_energy[:,1]

def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')


window_size = 100

rewards_smooth = moving_average(rewards, window_size)
tasks_dropped_smooth = moving_average(tasks_dropped, window_size)
# for throughput in throughputs:
#     throughputs_normalized.append(interp(throughput,[0,max(throughputs)],[0,7]))

# energy_normalized = []

# for energy in energies:
#     energy_normalized.append(interp(energy,[0,max(energies)],[0,55]))

# timesteps = []
# count = 0
# for task_dropped in tasks_dropped:
#     timesteps.append(count)
#     count+=1

# figure, axis = plt.subplots(2,1)

# Energy Efficiency Energy and Throughput
# axis[0].plot(timesteps[window_size-1:], battery_energy_rewards_smooth)
# axis[0].set_ylabel('Battery Energy Reward')
# axis[0].set_xlabel('Timestep(t)')
# axis[0].grid()

# axis[1].plot(timesteps[window_size-1:], tasks_dropped_smooth)
# axis[1].set_ylabel('Number of Tasks Dropped')
# axis[1].set_xlabel('Timestep(t)')
# axis[1].grid()

# axis[2].plot(timesteps, throughputs_normalized)
# axis[2].set_ylabel('Throughput (Mbits/s)')
# axis[2].set_xlabel('Timestep(t)')

# def moving_average(data, window_size):
#     """Compute the moving average of data."""
#     weights = np.repeat(1.0, window_size) / window_size
#     return np.convolve(data, weights, 'valid')

# window_size = 100

# rewards_1_users_smooth = moving_average(rewards_1_users, window_size)
# rewards_3_users_smooth = moving_average(rewards_3_users, window_size)
# rewards_7_users_smooth = moving_average(rewards_7_users, window_size)
# rewards_9_users_smooth = moving_average(rewards_9_users, window_size)


plt.plot(timesteps[window_size-1:], rewards_smooth, color="green", label="1 User")
# plt.plot(timesteps_3_users[window_size-1:], rewards_3_users_smooth, color="blue", label='3 Users')
# plt.plot(timesteps_7_users[window_size-1:], rewards_7_users_smooth, color="red", label='7 Users')
# plt.plot(timesteps_9_users[window_size-1:], rewards_9_users_smooth, color="black", label='9 Users')

# plt.xlabel("Timestep(t)")
# plt.ylabel("System Reward($\mathcal{R}$)")
# plt.legend(["1 User","3 Users", "7 Users", "9 Users"], loc="upper left")
#plt.grid()

plt.tight_layout()

plt.show()



