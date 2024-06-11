import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


inf_throughput_1_users_0_2 = np.load('inf_throughput_1_users_0_2.npy')
inf_throughput_1_users_0_4 = np.load('inf_throughput_1_users_0_4.npy')
inf_throughput_1_users_0_6 = np.load('inf_throughput_1_users_0_6.npy')
inf_throughput_1_users_0_8 = np.load('inf_throughput_1_users_0_8.npy')
inf_throughput_1_users_1 = np.load('inf_throughput_1_users_1.npy')

inf_throughput_3_users_0_2 = np.load('inf_throughput_3_users_0_2.npy')
inf_throughput_3_users_0_4 = np.load('inf_throughput_3_users_0_4.npy')
inf_throughput_3_users_0_6 = np.load('inf_throughput_3_users_0_6.npy')
inf_throughput_3_users_0_8 = np.load('inf_throughput_3_users_0_8.npy')
inf_throughput_3_users_1 = np.load('inf_throughput_3_users_1.npy')


p = [0.2,0.4,0.6,0.8,1]
throughputs_1_users = [inf_throughput_1_users_0_2,inf_throughput_1_users_0_4,inf_throughput_1_users_0_6,inf_throughput_1_users_0_8,inf_throughput_1_users_1]
throughputs_3_users = [inf_throughput_3_users_0_2,inf_throughput_3_users_0_4,inf_throughput_3_users_0_6,inf_throughput_3_users_0_8,inf_throughput_3_users_1]

plt.plot(p,throughputs_1_users,'--', marker='*', ms = 10,color="green")
plt.plot(p,throughputs_3_users,'--', marker='*', ms = 10,color="blue")

plt.xlabel("Probabilty of generating task")
plt.ylabel("Throughput")
#plt.legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")
#plt.grid()

plt.tight_layout()

plt.show()
