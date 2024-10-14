import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

users = [1,3,5,7,9]
rewards_TD3 = [3491669445.086330,6213448854.972728,8299887306.846853,10747468210.837811,13325714188.122597]
rewards_DDPG = [2616310443.630260 (1 users), 3489206827.739998 (3 users), 4643189572.781940 (5 users), 7485253962.268092 (9 users)]

plt.plot(users, rewards_TD3, color="green", marker='s')
plt.plot(users, rewards_TD3, color="green", marker='s')
# plt.plot(new_timesteps[window_size-1:], rewards_7_users_smooth[0:len_timesteps], color="brown", label='3 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_3_users_smooth[0:len_timesteps], color="blue", label='7 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Number of Users")
plt.ylabel("System Reward")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users"], loc="upper left")
plt.grid()

plt.tight_layout()
plt.show()



