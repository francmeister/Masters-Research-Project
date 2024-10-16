import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

users = [1,3,5,7,9]
rewards = [22866024.332961,22106013.889061,22773363.595256,23650460.309246,25036270.762014]


plt.plot(users, rewards, color="green", marker='s')
# plt.plot(new_timesteps[window_size-1:], rewards_7_users_smooth[0:len_timesteps], color="brown", label='3 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_3_users_smooth[0:len_timesteps], color="blue", label='7 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Number of Users")
plt.ylabel("Throughput")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users"], loc="upper left")
plt.grid()

plt.tight_layout()
plt.show()



