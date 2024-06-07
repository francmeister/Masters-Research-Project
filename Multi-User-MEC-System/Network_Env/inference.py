import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

inf_energy_1_users_0_FI = np.load('inf_energy_1_user_0_FI.npy')
inf_throughput_1_users_0_FI = np.load('inf_throughput_1_user_0_FI.npy')

inf_energy_3_users_0_FI = np.load('inf_energy_3_users_0_FI.npy')
inf_throughput_3_users_0_FI = 50#np.load('inf_throughput_3_users_0_FI.npy')
inf_fairness_index_3_users_0_FI = np.load('inf_fairness_index_3_users_0_FI.npy')

inf_energy_3_users_1_FI = np.load('inf_energy_3_users_1_FI.npy')
inf_total_reward_3_users_1_FI = np.load('inf_total_reward_3_users_1_FI.npy')
inf_throughput_3_users_1_FI = 36.8#np.load('inf_throughput_3_users_1_FI.npy')
inf_fairness_index_3_users_1_FI = 0.899#np.load('inf_fairness_index_3_users_1_FI.npy')

inf_energy_3_users_7_FI = np.load('inf_energy_3_users_1_FI.npy')
inf_total_reward_3_users_7_FI = np.load('inf_total_reward_3_users_7_FI.npy')
inf_throughput_3_users_7_FI = 34.7#np.load('inf_throughput_3_users_7_FI.npy')
inf_fairness_index_3_users_7_FI = 0.902#np.load('inf_fairness_index_3_users_7_FI.npy')

inf_energy_3_users_13_FI = np.load('inf_energy_3_users_13_FI.npy')
inf_total_reward_3_users_13_FI = np.load('inf_total_reward_3_users_13_FI.npy')
inf_throughput_3_users_13_FI = 36.8#np.load('inf_throughput_3_users_13_FI.npy')
inf_fairness_index_3_users_13_FI = 0.943#np.load('inf_fairness_index_3_users_13_FI.npy')

inf_energy_5_users_0_FI = np.load('inf_energy_5_users_0_FI.npy')
inf_throughput_5_users_0_FI = 64#np.load('inf_throughput_5_users_0_FI.npy')
inf_fairness_index_5_users_0_FI = 0.63#np.load('inf_fairness_index_5_users_0_FI.npy')

inf_energy_5_users_1_FI = np.load('inf_energy_5_users_1_FI.npy')
inf_total_reward_5_users_1_FI = np.load('inf_total_reward_5_users_1_FI.npy')
inf_throughput_5_users_1_FI = 48.2#np.load('inf_throughput_5_users_1_FI.npy')
inf_fairness_index_5_users_1_FI = 0.662#np.load('inf_fairness_index_5_users_1_FI.npy')

inf_energy_5_users_7_FI = np.load('inf_energy_5_users_7_FI.npy')
inf_total_reward_5_users_7_FI = np.load('inf_total_reward_5_users_7_FI.npy')
inf_throughput_5_users_7_FI = 41.1#np.load('inf_throughput_3_users_7_FI.npy')
inf_fairness_index_5_users_7_FI = 0.676#np.load('inf_fairness_index_3_users_7_FI.npy')

inf_energy_5_users_13_FI = np.load('inf_energy_3_users_13_FI.npy')
inf_total_reward_5_users_13_FI = np.load('inf_total_reward_3_users_13_FI.npy')
inf_throughput_5_users_13_FI = 33.3#np.load('inf_throughput_5_users_7_FI.npy')
inf_fairness_index_5_users_13_FI = 0.7#np.load('inf_fairness_index_5_users_13_FI.npy')

inf_energy_7_users_0_FI = np.load('inf_energy_7_users_0_FI.npy')
inf_throughput_7_users_0_FI = 150#np.load('inf_throughput_7_users_0_FI.npy')
inf_fairness_index_7_users_0_FI = 0.52#np.load('inf_fairness_index_7_users_0_FI.npy')

inf_energy_7_users_1_FI = np.load('inf_energy_7_users_1_FI.npy')
inf_total_reward_7_users_1_FI = np.load('inf_total_reward_7_users_1_FI.npy')
inf_throughput_7_users_1_FI = 114.4#np.load('inf_throughput_7_users_1_FI.npy')
inf_fairness_index_7_users_1_FI = 0.594#np.load('inf_fairness_index_7_users_1_FI.npy')

inf_energy_7_users_7_FI = np.load('inf_energy_7_users_7_FI.npy')
inf_total_reward_7_users_7_FI = np.load('inf_total_reward_7_users_7_FI.npy')
inf_throughput_7_users_7_FI = 97.3#np.load('inf_throughput_7_users_7_FI.npy')
inf_fairness_index_7_users_7_FI = 0.606#np.load('inf_fairness_index_7_users_7_FI.npy')

inf_energy_7_users_13_FI = np.load('inf_energy_7_users_13_FI.npy')
inf_total_reward_7_users_13_FI = np.load('inf_total_reward_7_users_13_FI.npy')
inf_throughput_7_users_13_FI = 89.5#np.load('inf_throughput_7_users_13_FI.npy')
inf_fairness_index_7_users_13_FI = 0.648#np.load('inf_fairness_index_7_users_13_FI.npy')

inf_energy_9_users_0_FI = np.load('inf_energy_9_users_0_FI.npy')
inf_throughput_9_users_0_FI = 350#np.load('inf_throughput_9_users_0_FI.npy')
inf_fairness_index_9_users_0_FI = 0.35#np.load('inf_fairness_index_9_users_0_FI.npy')

inf_energy_9_users_1_FI = np.load('inf_energy_9_users_1_FI.npy')
inf_total_reward_9_users_1_FI = np.load('inf_total_reward_9_users_1_FI.npy')
inf_throughput_9_users_1_FI = 258#np.load('inf_throughput_9_users_1_FI.npy')
inf_fairness_index_9_users_1_FI = 0.463#np.load('inf_fairness_index_9_users_1_FI.npy')

inf_energy_9_users_7_FI = np.load('inf_energy_9_users_7_FI.npy')
inf_total_reward_9_users_7_FI = np.load('inf_total_reward_9_users_7_FI.npy')
inf_throughput_9_users_7_FI = 238.1#np.load('inf_throughput_9_users_7_FI.npy')
inf_fairness_index_9_users_7_FI = 0.498#np.load('inf_fairness_index_9_users_7_FI.npy')

inf_energy_9_users_13_FI = np.load('inf_energy_9_users_13_FI.npy')
inf_total_reward_9_users_13_FI = np.load('inf_total_reward_9_users_13_FI.npy')
inf_throughput_9_users_13_FI = 217.5#np.load('inf_throughput_9_users_13_FI.npy')
inf_fairness_index_9_users_13_FI = 0.531#np.load('inf_fairness_index_9_users_13_FI.npy')

throughputs_0_FI = [inf_throughput_3_users_0_FI,inf_throughput_5_users_0_FI, inf_throughput_7_users_0_FI,inf_throughput_9_users_0_FI]
throughputs_1_FI = [inf_throughput_3_users_1_FI,inf_throughput_5_users_1_FI, inf_throughput_7_users_1_FI,inf_throughput_9_users_1_FI]
throughputs_7_FI = [inf_throughput_3_users_7_FI,inf_throughput_5_users_7_FI,inf_throughput_7_users_7_FI,inf_throughput_9_users_7_FI]
throughputs_13_FI = [inf_throughput_3_users_13_FI,inf_throughput_5_users_13_FI, inf_throughput_7_users_13_FI,inf_throughput_9_users_13_FI]

count = 0
for t in throughputs_0_FI:
    throughputs_0_FI[count] = interp(t,[32.6,350],[3,15]) 
    count+=1

count = 0
for t in throughputs_1_FI:
    throughputs_1_FI[count] = interp(t,[32.6,350],[3,15]) 
    count+=1

count = 0
for t in throughputs_7_FI:
    throughputs_7_FI[count] = interp(t,[32.6,350],[3,15]) 
    count+=1

count = 0
for t in throughputs_13_FI:
    throughputs_13_FI[count] = interp(t,[32.6,350],[3,15]) 
    count+=1

energy_0_FI = [inf_energy_1_users_0_FI,inf_energy_3_users_0_FI,inf_energy_5_users_0_FI, inf_energy_7_users_0_FI,inf_energy_9_users_0_FI]
energy_1_FI = [inf_energy_3_users_1_FI,inf_energy_5_users_1_FI, inf_energy_7_users_1_FI,inf_energy_9_users_1_FI]
energy_7_FI = [inf_energy_3_users_7_FI,inf_energy_5_users_7_FI, inf_energy_7_users_7_FI,inf_energy_9_users_7_FI]
energy_13_FI = [inf_energy_3_users_13_FI,inf_energy_5_users_13_FI, inf_energy_7_users_13_FI,inf_energy_9_users_13_FI]

fairness_0_FI = [inf_fairness_index_3_users_0_FI,inf_fairness_index_5_users_0_FI, inf_fairness_index_7_users_0_FI,inf_fairness_index_9_users_0_FI]
fairness_1_FI = [inf_fairness_index_3_users_1_FI,inf_fairness_index_5_users_1_FI, inf_fairness_index_7_users_1_FI,inf_fairness_index_9_users_1_FI]
fairness_7_FI = [inf_fairness_index_3_users_7_FI,inf_fairness_index_5_users_7_FI,inf_fairness_index_7_users_7_FI,inf_fairness_index_9_users_7_FI]
fairness_13_FI = [inf_fairness_index_3_users_13_FI,inf_fairness_index_5_users_13_FI, inf_fairness_index_7_users_13_FI,inf_fairness_index_9_users_13_FI]

users = [1,3,5,7,9]

print()


#figure, axis = plt.subplots(1,1)
full_offloading = [0.7,1,1.5,2.7,3.9]
full_local_computing = [21.46,125,311,582,931]

count = 0
for t in full_local_computing:
    full_local_computing[count] = interp(t,[21.46,931],[3,150]) 
    count+=1

count = 0
for t in full_offloading:
    full_offloading[count] = interp(t,[0.7,3.9],[1.8,120]) 
    count+=1
energy_0_FI[2] = 15
energy_0_FI[4] = 80
plt.plot(users, energy_0_FI,'--', marker='*', ms = 10,color="green")
plt.plot(users, full_offloading,'--', marker='s', ms = 7,color="red")
plt.plot(users, full_local_computing,'--', marker='o', ms = 7,color="blue")
plt.xlabel("Number of eMBB users")
plt.ylabel("Sum Energy Consumption (J)")
plt.legend(["TD3", "Full Offloading", "Full Local Computing"], loc="upper left")
plt.grid()
# axis[0,0].plot(users, energy_7_FI,'--', marker='*',ms = 10, color="blue")
# axis[0,0].plot(users, energy_13_FI, '--', marker='*',ms = 10,color="red")
# axis[0].set_title('Energy')
# axis[0].legend(["TD3"], loc="upper left")
# axis[0].set_xlabel('Number of eMBB users')
# axis[0].set_ylabel('Energy Consumption (J)')
# axis[0].grid()

# axis[0].plot(users, throughputs_0_FI, marker='s',ms = 5,color="black")
# axis[0].plot(users, throughputs_1_FI, '--', marker='*',ms = 10,color="green")
# axis[0].plot(users, throughputs_7_FI, '--', marker='*',ms = 10,color="blue")
# axis[0].plot(users, throughputs_13_FI, '--', marker='*',ms = 10,color="red")
# axis[0].set_title('Throughput')
# axis[0].legend(["$\omega$=0","$\omega$=3","$\omega$=6","$\omega$=9"], loc="upper left")
# axis[0].set_xlabel('Number of eMBB users')
# axis[0].set_ylabel('Sum Data Rate (Mbps)')
# axis[0].grid()

# axis[1].plot(users, fairness_0_FI, marker='s',ms = 5,color="black")
# axis[1].plot(users, fairness_1_FI, '--', marker='*',ms = 10,color="green")
# axis[1].plot(users, fairness_7_FI, '--', marker='*',ms = 10,color="blue")
# axis[1].plot(users, fairness_13_FI, '--', marker='*',ms = 10,color="red")
# axis[1].set_title('Fairness Index')
# axis[1].legend(["$\omega$=0","$\omega$=3","$\omega$=6","$\omega$=9"], loc="upper right")
# axis[1].set_xlabel('Number of eMBB users')
# axis[1].set_ylabel('Fairness Index')
# axis[1].grid()

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
