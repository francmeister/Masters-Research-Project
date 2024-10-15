import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

users = [1,3,5,7,9]
rewards_TD3 = [51.000000,153.036848,1651.543618,2728.945541,6260.636093]
rewards_FL = [524198,1177489,1793394,2675364,3339631]
rewards_FO = [51.0,159.38168788307476,541.4518712558075,2462.1473670801124,5850.867704483366]


plt.plot(users, rewards_TD3, color="green", marker='s')
#plt.plot(users, rewards_FL, color="red", marker='s')
plt.plot(users, rewards_FO, color="blue", marker='s')
# plt.plot(new_timesteps[window_size-1:], rewards_7_users_smooth[0:len_timesteps], color="brown", label='3 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_3_users_smooth[0:len_timesteps], color="blue", label='7 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Number of Users")
plt.ylabel("Delay (ms)")
plt.legend(["TD3","Full Local Computing", "Full Offloading"], loc="upper left")
plt.grid()

plt.tight_layout()
plt.show()



