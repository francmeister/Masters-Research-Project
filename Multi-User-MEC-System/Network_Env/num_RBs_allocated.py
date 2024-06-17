import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
from scipy.interpolate import make_interp_spline, BSpline
#inf_throughput_7_users_1 = np.load('inf_throughput_7_users_1.npy')


embb_users = [1,3,5,7,9]

num_RBs_allocated_TD3 = []
# plt.plot(embb_users,throughputs_0_2,'--', marker='*', ms = 10,color="green")
# plt.plot(embb_users,throughputs_0_4,'--', marker='*', ms = 10,color="blue")
# plt.plot(embb_users,throughputs_0_6,'--', marker='*', ms = 10,color="red")
# plt.plot(embb_users,throughputs_0_8,'--', marker='*', ms = 10,color="black")
# plt.plot(embb_users,throughputs_1,'--', marker='*', ms = 10,color="grey")
plt.plot(embb_users,num_RBs_allocated_TD3,'--', marker='*', ms = 10,color="brown")
plt.plot(embb_users,random_actions_1,'--', marker='s', ms = 10,color="green")
# #plt.plot(embb_users_,throughputs_6_urllc_users_,'--', ms = 10,color="brown", linestyle='-')

# urllc_users = [1,2,3,4,5,6]
# plt.plot(urllc_users,all_throughput[:,0],'--', marker='*', ms = 10,color="green")
# plt.plot(urllc_users,all_throughput[:,1],'--', marker='*', ms = 10,color="blue")
# plt.plot(urllc_users,all_throughput[:,2],'--', marker='*', ms = 10,color="red")
# plt.plot(urllc_users,all_throughput[:,3],'--', marker='*', ms = 10,color="black")
# plt.plot(urllc_users,all_throughput[:,4],'--', marker='*', ms = 10,color="grey")
# plt.plot(urllc_users,all_throughput[:,5],'--', marker='*', ms = 10,color="brown")


plt.xlabel("Number of eMBB Users")
plt.ylabel("eMBB Users Sum Data Rate (Mbps)")
plt.legend(["p = 1, TD3-based Action Generation","p = 1, Random Action Generation"], loc="upper left")

# plt.xlabel("Number of eMBB Users")
# plt.ylabel("eMBB Users Sum Data Rate (Mbps)")
# plt.legend(["p = 0.2","p = 0.4","p = 0.6", "p = 0.8", "p = 1"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
