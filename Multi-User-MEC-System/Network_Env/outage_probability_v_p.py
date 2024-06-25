import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


p = [0.2,0.4,0.6,0.8,1]
num_embb_users = [1,3,5,9]
op_2_embb_users = [0.000673,0.001495,0.002223, 0.004759, 0.009209]
op_1_embb_users = [0.000456,0.001116,0.002882,0.005352,0.008545]
op_3_embb_users = [0.050118,0.076384,0.114732,0.155384,0.211177]
op_5_embb_users = [0.102644,0.184315,0.269739,0.386359,0.490675]
op_9_embb_users = [0.167373,0.317005,0.413083,0.572011,0.683124]#[0.243622,0.409520,0.503342,0.643591,0.728535]#[0.167373,0.317005,0.413083,0.572011,0.683124]
op_7_embb_users = [0.128614,0.232231,0.341670,0.482004,0.609616]

all_users = [op_1_embb_users,op_3_embb_users,op_5_embb_users,op_9_embb_users]
np_array = np.array(all_users)
print(np_array)
print(np_array[:,0])

#op_9_embb_users_random = [0.11564682046672566,0.21282293090496576,0.3621023035655675,0.4657099822521426,0.6018962972432126]
op_1_embb_users_random = [0.1810368698570475,0.24275875582542297,0.35231021777398847,0.4683824466556121,0.5849676441638699]


#plt.plot(p,op_1_embb_users,'--', marker='*', ms = 10,color="green")
plt.plot(p,op_3_embb_users,'--', marker='*', ms = 10,color="green")
# plt.plot(p,op_5_embb_users,'--', marker='*', ms = 10,color="red")
# plt.plot(p,op_7_embb_users,'--', marker='*', ms = 10,color="grey")
# plt.plot(p,op_9_embb_users,'--', marker='*', ms = 10,color="black")
plt.plot(p,op_1_embb_users_random,'--', marker='s', ms = 10,color="black")

# plt.plot(num_embb_users,np_array[:,0],'--', marker='*', ms = 10,color="green")
# plt.plot(num_embb_users,np_array[:,1],'--', marker='*', ms = 10,color="blue")
# plt.plot(num_embb_users,np_array[:,2],'--', marker='*', ms = 10,color="red")
# #plt.plot(p,op_5_embb_users,'--', marker='*', ms = 10,color="red")
# plt.plot(num_embb_users,np_array[:,3],'--', marker='*', ms = 10,color="black")
# plt.plot(num_embb_users,np_array[:,4],'--', marker='*', ms = 10,color="pink")


plt.xlabel("URLLC Users' Probabilty of Generating a Task")
#plt.xlabel("Number of eMBB Users")
plt.ylabel("URLLC Users Outage Probability ($\phi$)")
#plt.legend(["1 eMBB User", "3 eMBB Users", "5 eMBB Users", "7 eMBB Users", "9 eMBB Users"], loc="upper left")
plt.legend(["TD3-based Action Generation, 6 URLLC Users", "Random Action Generation, 6 URLLC Users"], loc="upper left")
#plt.legend(["p = 0.2", "p = 0.4", "p = 0.6", "p = 0.8", "p = 1"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
