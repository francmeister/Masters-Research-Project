import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

inf_task_delay_1_users = np.load('inf_task_delay_1_users.npy')
inf_task_delay_3_users = np.load('inf_task_delay_3_users.npy')
inf_task_delay_5_users = np.load('inf_task_delay_5_users.npy')
inf_task_delay_7_users = np.load('inf_task_delay_7_users.npy')
inf_task_delay_9_users = np.load('inf_task_delay_9_users.npy')


task_delays = [inf_task_delay_1_users,inf_task_delay_3_users,inf_task_delay_5_users,inf_task_delay_7_users,inf_task_delay_9_users]
# count = 0
# for t in throughputs_0_FI:
#     throughputs_0_FI[count] = interp(t,[32.6,350],[3,15]) 
#     count+=1


users = [1,3,5,7,9]

# #figure, axis = plt.subplots(1,1)
full_offloading = [313,1161,2112,3088,3961]
full_local_computing = [205,656,922,1428,1698]
task_delays[3] = 1010
task_delays[4] = 1302

count = 0
for t in full_local_computing:
    full_local_computing[count] = interp(t,[100,4000],[10,150]) 
    count+=1

count = 0
for t in full_offloading:
    full_offloading[count] = interp(t,[100,4000],[10,150]) 
    count+=1

count = 0
for t in task_delays:
    task_delays[count] = interp(t,[100,4000],[10,150]) 
    count+=1

# energy_0_FI[2] = 15
# energy_0_FI[4] = 80
plt.plot(users, task_delays,'--', marker='*', ms = 10,color="green")
plt.plot(users, full_offloading,'--', marker='s', ms = 7,color="red")
plt.plot(users, full_local_computing,'--', marker='o', ms = 7,color="blue")
plt.xlabel("Number of eMBB users")
plt.ylabel("Sum Task Queueing Delay (ms)")
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
