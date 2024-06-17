import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
from scipy.interpolate import make_interp_spline, BSpline
#inf_throughput_7_users_1 = np.load('inf_throughput_7_users_1.npy')


embb_users = [1,3,5,7,9]

#embb_users = np.array([1,3,5,7,9])

throughputs_0_2 = [11.938769,23.179472,46.643599,82.467801,101.967332]
throughputs_0_4 = [11.905622,22.511795,45.267431,80.296416,99.374557]
throughputs_0_6 = [11.834773,21.797246,44.439722,78.262687,96.254796]
throughputs_0_8 = [11.773056,21.129006,43.048654,76.014374,93.573598]
throughputs_1 = [10.466878,20.470109,41.679597,73.845885,91.101242]

random_actions_1 = [10.962542470271405, 22.33687042972403, 33.6112789077141, 45.07090661316891, 56.49095765763066]

all_throughput = np.array([throughputs_0_2,throughputs_0_4,throughputs_0_6,throughputs_0_8,throughputs_1])
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
for t in throughputs_0_2:
    throughputs_0_2[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in throughputs_0_4:
    throughputs_0_4[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1



count = 0
for t in throughputs_0_6:
    throughputs_0_6[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in throughputs_0_8:
    throughputs_0_8[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

count = 0
for t in throughputs_1:
    throughputs_1[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1


# count = 0
# for t in throughputs_1:
#     throughputs_0_2[count] = interp(t,[min_t,max_t],[3,15]) 
#     count+=1

count = 0
for t in random_actions_1:
    random_actions_1[count] = interp(t,[min_t,max_t],[3,15]) 
    count+=1

# plt.plot(embb_users,throughputs_0_2,'--', marker='*', ms = 10,color="green")
# plt.plot(embb_users,throughputs_0_4,'--', marker='*', ms = 10,color="blue")
# plt.plot(embb_users,throughputs_0_6,'--', marker='*', ms = 10,color="red")
# plt.plot(embb_users,throughputs_0_8,'--', marker='*', ms = 10,color="black")
# plt.plot(embb_users,throughputs_1,'--', marker='*', ms = 10,color="grey")
plt.plot(embb_users,throughputs_1,'--', marker='*', ms = 10,color="brown")
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
