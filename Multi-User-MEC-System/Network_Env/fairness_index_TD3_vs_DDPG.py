import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

# A total of 12 urllc users
fairness_index_TD3 = [0.914771, 0.819318, 0.593818] #(3,5,7,9 users) 0.810910 = 7 Users FI
fairness_index_DDPG = [0.977384,0.898898, 0.765581] #(3,5,9 users)
num_users = [3,5,9]

plt.plot(num_users, fairness_index_TD3, color='green', marker='s')
plt.plot(num_users, fairness_index_DDPG, color='red', marker='s')


plt.xlabel("Number of Users")
plt.ylabel("Fairness Index")
plt.legend(["TD3", "DDPG"], loc="upper right")

#plt.xlabel("Timestep(t)")
#plt.ylabel("Fairness Index")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users", "11 Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
