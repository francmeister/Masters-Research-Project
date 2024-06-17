import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
from scipy.interpolate import make_interp_spline, BSpline
#inf_throughput_7_users_1 = np.load('inf_throughput_7_users_1.npy')


embb_users = [1,3,5,7,9]

#embb_users = np.array([1,3,5,7,9])

throughputs_0_2 = []
throughputs_0_4 = []
throughputs_0_6 = []
throughputs_0_8 = []
throughputs_1 = []
random_actions_6_urllc_users = []

all_throughput = np.array([throughputs_0_2,throughputs_0_4,throughputs_0_6,throughputs_0_8,throughputs_1,random_actions_6_urllc_users])
print(all_throughput)
# embb_users_ = np.linspace(embb_users.min(), embb_users.max(), 200) 

# #define spline
# spl = make_interp_spline(embb_users, throughputs_6_urllc_users, k=3)
# throughputs_6_urllc_users_ = spl(embb_users_)

min_t = np.amin(all_throughput)
max_t = np.amax(all_throughput)
print(min_t)
print(max_t)

for r in range(0,len(all_throughput[:,0])):
    for c in range(0,len(all_throughput[0,:])):
        all_throughput[r,c] = interp(all_throughput[r,c],[min_t,max_t],[3,15])

print(all_throughput)        
count = 0
for t in throughputs_1_urllc_users:
    throughputs_1_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in throughputs_2_urllc_users:
    throughputs_2_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1



count = 0
for t in throughputs_3_urllc_users:
    throughputs_3_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in throughputs_4_urllc_users:
    throughputs_4_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in throughputs_5_urllc_users:
    throughputs_5_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1


count = 0
for t in throughputs_6_urllc_users:
    throughputs_6_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in random_actions_6_urllc_users:
    random_actions_6_urllc_users[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

# plt.plot(embb_users,throughputs_1_urllc_users,'--', marker='*', ms = 10,color="green")
# plt.plot(embb_users,throughputs_2_urllc_users,'--', marker='*', ms = 10,color="blue")
# plt.plot(embb_users,throughputs_3_urllc_users,'--', marker='*', ms = 10,color="red")
# plt.plot(embb_users,throughputs_4_urllc_users,'--', marker='*', ms = 10,color="black")
# plt.plot(embb_users,throughputs_5_urllc_users,'--', marker='*', ms = 10,color="grey")
plt.plot(embb_users,throughputs_6_urllc_users,'--', marker='*', ms = 10,color="brown")
plt.plot(embb_users,random_actions_6_urllc_users,'--', marker='s', ms = 10,color="green")
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
plt.legend(["TD3-based Action Generation with 6 URLLC Users","Random Action Generation with 6 URLLC Users"], loc="upper left")

# plt.xlabel("Number of URLLC Users")
# plt.ylabel("eMBB Users Sum Data Rate (Mbps)")
# plt.legend(["1 eMBB User","3 eMBB Users","5 eMBB Users", "7 eMBB Users", "9 eMBB Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
