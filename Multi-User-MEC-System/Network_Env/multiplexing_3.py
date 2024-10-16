import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

# A total of 12 urllc users
nine_embb_user_TD3_throughput = [25868674.065489,25337179.268292, 24587164.554136, 24008365.448069, 23340605.883614]
nine_embb_user_DDPG_throughput = [21502714.240237,21074582.007114, 20504106.694446, 19960857.639323, 19422672.062589]
num_urllc_users = [3,5,7,9, 11]

plt.plot(num_urllc_users, nine_embb_user_TD3_throughput, color='green', marker='s')
plt.plot(num_urllc_users, nine_embb_user_DDPG_throughput, color='red', marker='s')


plt.xlabel("Number of URLLC Users")
plt.ylabel("Sum Data Rate (bits/s)")
plt.legend(["TD3", "DDPG"], loc="upper right")

#plt.xlabel("Timestep(t)")
#plt.ylabel("Fairness Index")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users", "11 Users"], loc="upper left")
plt.grid()
plt.tight_layout()
plt.show()
