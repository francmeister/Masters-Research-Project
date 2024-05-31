import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


inf_energy_3_users_1_FI = np.load('inf_energy_3_users_1_FI.npy')
inf_total_reward_3_users_1_FI = np.load('inf_total_reward_3_users_1_FI.npy')
inf_throughput_3_users_1_FI = np.load('inf_throughput_3_users_1_FI.npy')
inf_fairness_index_3_users_1_FI = np.load('inf_fairness_index_3_users_1_FI.npy')

inf_energy_3_users_7_FI = np.load('inf_energy_3_users_1_FI.npy')
inf_total_reward_3_users_7_FI = np.load('inf_total_reward_3_users_7_FI.npy')
inf_throughput_3_users_7_FI = np.load('inf_throughput_3_users_7_FI.npy')
inf_fairness_index_3_users_7_FI = np.load('inf_fairness_index_3_users_7_FI.npy')

inf_energy_3_users_13_FI = np.load('inf_energy_3_users_13_FI.npy')
inf_total_reward_3_users_13_FI = np.load('inf_total_reward_3_users_13_FI.npy')
inf_throughput_3_users_13_FI = np.load('inf_throughput_3_users_13_FI.npy')
inf_fairness_index_3_users_13_FI = np.load('inf_fairness_index_3_users_13_FI.npy')

inf_energy_5_users_1_FI = np.load('inf_energy_5_users_1_FI.npy')
inf_total_reward_5_users_1_FI = np.load('inf_total_reward_5_users_1_FI.npy')
inf_throughput_5_users_1_FI = np.load('inf_throughput_5_users_1_FI.npy')
inf_fairness_index_5_users_1_FI = np.load('inf_fairness_index_5_users_1_FI.npy')

inf_energy_5_users_7_FI = np.load('inf_energy_5_users_7_FI.npy')
inf_total_reward_5_users_7_FI = np.load('inf_total_reward_5_users_7_FI.npy')
inf_throughput_5_users_7_FI = np.load('inf_throughput_3_users_7_FI.npy')
inf_fairness_index_5_users_7_FI = np.load('inf_fairness_index_3_users_7_FI.npy')

inf_energy_5_users_13_FI = np.load('inf_energy_3_users_13_FI.npy')
inf_total_reward_5_users_13_FI = np.load('inf_total_reward_3_users_13_FI.npy')
inf_throughput_5_users_13_FI = np.load('inf_throughput_5_users_7_FI.npy')
inf_fairness_index_5_users_13_FI = np.load('inf_fairness_index_5_users_13_FI.npy')

inf_energy_9_users_1_FI = np.load('inf_energy_9_users_1_FI.npy')
inf_total_reward_9_users_1_FI = np.load('inf_total_reward_9_users_1_FI.npy')
inf_throughput_9_users_1_FI = np.load('inf_throughput_9_users_1_FI.npy')
inf_fairness_index_9_users_1_FI = np.load('inf_fairness_index_9_users_1_FI.npy')

inf_energy_9_users_7_FI = np.load('inf_energy_9_users_7_FI.npy')
inf_total_reward_9_users_7_FI = np.load('inf_total_reward_9_users_7_FI.npy')
inf_throughput_9_users_7_FI = np.load('inf_throughput_9_users_7_FI.npy')
inf_fairness_index_9_users_7_FI = np.load('inf_fairness_index_9_users_7_FI.npy')

inf_energy_9_users_13_FI = np.load('inf_energy_9_users_13_FI.npy')
inf_total_reward_9_users_13_FI = np.load('inf_total_reward_9_users_13_FI.npy')
inf_throughput_9_users_13_FI = np.load('inf_throughput_9_users_13_FI.npy')
inf_fairness_index_9_users_13_FI = np.load('inf_fairness_index_9_users_13_FI.npy')

throughputs_1_FI = [inf_throughput_3_users_1_FI,inf_throughput_5_users_1_FI,inf_throughput_9_users_1_FI]
throughputs_7_FI = [inf_throughput_3_users_7_FI,inf_throughput_5_users_7_FI,inf_throughput_9_users_7_FI]
throughputs_13_FI = [inf_throughput_3_users_13_FI,inf_throughput_5_users_13_FI,inf_throughput_9_users_13_FI]

energy_1_FI = [inf_energy_3_users_1_FI,inf_energy_5_users_1_FI,inf_energy_9_users_1_FI]
energy_7_FI = [inf_energy_3_users_7_FI,inf_energy_5_users_7_FI,inf_energy_9_users_7_FI]
energy_13_FI = [inf_energy_3_users_13_FI,inf_energy_5_users_13_FI,inf_energy_9_users_13_FI]

fairness_1_FI = [inf_fairness_index_3_users_1_FI,inf_fairness_index_5_users_1_FI,inf_fairness_index_9_users_1_FI]
fairness_7_FI = [inf_fairness_index_3_users_7_FI,inf_fairness_index_5_users_7_FI,inf_fairness_index_9_users_7_FI]
fairness_13_FI = [inf_fairness_index_3_users_13_FI,inf_fairness_index_5_users_13_FI,inf_fairness_index_9_users_13_FI]

users = [3,5,9]

print()


figure, axis = plt.subplots(2,2)

axis[0,0].plot(users, energy_1_FI,'--', marker='*', ms = 10,color="green")
axis[0,0].plot(users, energy_7_FI,'--', marker='*',ms = 10, color="blue")
axis[0,0].plot(users, energy_13_FI, '--', marker='*',ms = 10,color="red")
axis[0,0].set_title('Energy')
axis[0,0].legend(["FI=1","FI=7","FI=13"], loc="upper left")
axis[0,0].grid()

axis[0,1].plot(users, throughputs_1_FI, '--', marker='*',ms = 10,color="green")
axis[0,1].plot(users, throughputs_7_FI, '--', marker='*',ms = 10,color="blue")
axis[0,1].plot(users, throughputs_13_FI, '--', marker='*',ms = 10,color="red")
axis[0,1].set_title('Throughput')
axis[0,1].legend(["FI=1","FI=7","FI=13"], loc="upper left")
axis[0,1].grid()

axis[1,0].plot(users, fairness_1_FI, '--', marker='*',ms = 10,color="green")
axis[1,0].plot(users, fairness_7_FI, '--', marker='*',ms = 10,color="blue")
axis[1,0].plot(users, fairness_13_FI, '--', marker='*',ms = 10,color="red")
axis[1,0].set_title('Fairness Index')
axis[1,0].legend(["FI=1","FI=7","FI=13"], loc="upper right")
axis[1,0].grid()

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

plt.tight_layout()

plt.show()
