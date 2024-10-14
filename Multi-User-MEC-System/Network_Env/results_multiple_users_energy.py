import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

users = [1,3,5,7,9]
rewards_TD3 = [0.002048,0.003279,0.001300,0.005962,0.002270]
rewards_FL = [0.06352791789021443,0.1901851867295864,0.31696066933398453,0.4436985078017845,0.5704175591985978]
rewards_FO = [0.00016040274177026443,0.0001712125556318549,0.00016725729547800806,0.00016565410259720265,0.00016701681163379693]



plt.plot(users, rewards_TD3, color="green", marker='s')
plt.plot(users, rewards_FL, color="blue", marker='s')
plt.plot(users, rewards_FO, color="red", marker='s')
# plt.plot(new_timesteps[window_size-1:], rewards_7_users_smooth[0:len_timesteps], color="brown", label='3 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_3_users_smooth[0:len_timesteps], color="blue", label='7 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_5_users_smooth[0:len_timesteps], color="grey", label='5 Users')
# plt.plot(new_timesteps[window_size-1:], rewards_9_users_smooth, color="red", label='9 Users')
#plt.plot(timesteps_11_users[window_size-1:], rewards_11_users_smooth, color="black", label='11 Users')

plt.xlabel("Number of Users")
plt.ylabel("Energy Consumption (J)")
plt.legend(["TD3","Full Local Computing", "Full Offloading"], loc="upper left")
plt.grid()

plt.tight_layout()
plt.show()



