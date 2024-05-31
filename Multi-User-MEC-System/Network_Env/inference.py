import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


inf_energy_3_users_1_FI = np.load('inf_energy_3_users_1_FI.npy')
inf_total_reward_3_users_1_FI = np.load('inf_total_reward_3_users_1_FI.npy')
inf_throughput_3_users_1_FI = np.load('inf_throughput_3_users_1_FI.npy')
inf_fairness_index_3_users_1_FI = np.load('inf_fairness_index_3_users_1_FI.npy')

print("inf energy: ", inf_energy_3_users_1_FI)
print("inf_total_reward: ", inf_total_reward_3_users_1_FI)
print("inf_throughput: ", inf_throughput_3_users_1_FI)
print("inf_fairness_index: ", inf_fairness_index_3_users_1_FI)



# figure, axis = plt.subplots(2,2)

# axis[0,0].plot(timesteps_1[window_size-1:], rewards_1_smooth, color="green")

# axis[0,0].set_title('Total Reward')
# axis[0,0].grid()

# axis[0,1].plot(timesteps_1[window_size-1:], fairness_index_1_smooth, color="green")

# axis[0,1].set_title('Fairness Index')
# axis[0,1].grid()

# axis[1,1].plot(timesteps_1[window_size-1:], throughput_1_smooth, color="green")

# axis[1,1].set_title('Throughput')
# axis[1,1].grid()

# axis[1,0].plot(timesteps_1[window_size-1:], energy_1_smooth, color="green")

# axis[1,0].set_title('Energy')
# axis[1,0].grid()

# plt.xlabel("Timestep(t)")
# #plt.ylabel("Reward")
# #plt.legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")
# #plt.grid()

# plt.tight_layout()

# plt.show()
