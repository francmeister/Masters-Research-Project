import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


outage_prabability = np.load('outage_probabilties.npy')
reliability_reward = np.load('urllc_reliability_reward_normalized.npy')
power_actions = np.load('power_actions.npy')
urllc_avg_rate = np.load('urllc_avg_rate.npy')

#print(outage_prabability)

timestep_rewards_energy_throughput = np.load('timestep_rewards_energy_throughput.npy')

timesteps = timestep_rewards_energy_throughput[:,0]

outage_constraint_matrix = []
outage_constraint = 0.05
for i in range(0,len(timesteps)):
    outage_constraint_matrix.append(outage_constraint)


print(outage_constraint_matrix)


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 200

outage_prabability_smooth = moving_average(outage_prabability, window_size)
reliability_reward_smooth = moving_average(reliability_reward, window_size)


# plt.plot(timesteps_1_users[window_size-1:], fairnes_index_1_Users_smooth, color="green", label="1 User")
# plt.plot(timesteps_3_users[window_size-1:], fairnes_index_3_Users_smooth, color="blue", label='3 Users')
# plt.plot(timesteps_5_users[window_size-1:], fairnes_index_5_Users_smooth, color="red", label='5 Users')
# plt.plot(timesteps_7_users[window_size-1:], fairnes_index_7_Users_smooth, color="purple", label='7 Users')
# plt.plot(timesteps_9_users[window_size-1:], fairnes_index_9_Users_smooth, color="grey", label='9 Users')
# plt.plot(timesteps_11_users[window_size-1:], fairnes_index_11_Users_smooth, color="black", label='11 Users')

new_timesteps = []
count = 0
for timestep in timesteps:
    new_timesteps.append(count)
    count+=1

plt.plot(new_timesteps[window_size-1:], outage_prabability_smooth, color='black')
plt.plot(new_timesteps,outage_constraint_matrix, '--', color='red', linewidth = 2.5)

plt.xlabel("Episodes")
plt.ylabel("URLLC Users Outage Probability ($\phi$)")
plt.legend(["$\phi$", "$\phi^{max}$"], loc="upper right")
# figure, axis = plt.subplots(4,1)

# axis[0].plot(timesteps[window_size-1:], outage_prabability_smooth)
# axis[0].set_title('Outage Probability')

# axis[1].plot(timesteps, reliability_reward)
# axis[1].set_title('Reliability Reward')

# axis[2].plot(timesteps, power_actions)
# axis[2].set_title('Power Actions')

# axis[3].plot(timesteps, urllc_avg_rate)
# axis[3].set_title('URLLC avg rate')

#plt.xlabel("Timestep(t)")
#plt.ylabel("Fairness Index")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users", "11 Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
