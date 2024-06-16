import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

#inf_throughput_7_users_1 = np.load('inf_throughput_7_users_1.npy')


embb_users = [1,3,5,7,9]


throughputs_1_urllc_users = [11.027825, 24.580076, 45.534253, 83.117946, 105.590108]
throughputs_2_urllc_users = [10.700383, 24.580076, 45.971143, 83.161688, 101.0385550]
throughputs_3_urllc_users = [10.285659, 22.840365, 44.017812, 78.369836, 99.003708]
throughputs_4_urllc_users = [10.081424, 21.883773, 42.848610, 78.892532, 96.340276]
throughputs_5_urllc_users = [9.871085, 21.137908, 41.807170, 75.538351, 91.620090]
throughputs_6_urllc_users = [10.056622, 20.470111, 41.679426, 73.846335, 91.118700]


plt.plot(embb_users,throughputs_1_urllc_users,'--', marker='*', ms = 10,color="green")
plt.plot(embb_users,throughputs_2_urllc_users,'--', marker='*', ms = 10,color="blue")
plt.plot(embb_users,throughputs_3_urllc_users,'--', marker='*', ms = 10,color="red")
plt.plot(embb_users,throughputs_4_urllc_users,'--', marker='*', ms = 10,color="black")
plt.plot(embb_users,throughputs_5_urllc_users,'--', marker='*', ms = 10,color="grey")
plt.plot(embb_users,throughputs_6_urllc_users,'--', marker='*', ms = 10,color="brown")


plt.xlabel("Number of eMBB Users")
plt.ylabel("eMBB Users Sum Throughput")
plt.legend(["1 URLLC User","2 URLLC Users","3 URLLC Users", "4 URLLC Users", "5 URLLC Users", "6 URLLC Users"], loc="upper left")
#plt.grid()

plt.tight_layout()

plt.show()
