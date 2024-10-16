import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

# A total of 12 urllc users
nine_embb_user_TD3_throughput = [25351095.047322,23801548.846854, 22345252.487799, 20776080.353478, 19232726.200073]
nine_embb_user_DDPG_throughput = [20917260.671260,19757380.357901, 18681259.867431, 17478832.637684, 16269957.156584]
prob_gen_task = [0.2,0.4,0.6,0.8, 1]

plt.plot(prob_gen_task, nine_embb_user_TD3_throughput, color='green', marker='s')
plt.plot(prob_gen_task, nine_embb_user_DDPG_throughput, color='red', marker='s')


plt.xlabel("URLLC Users' Probability of Generating a Task")
plt.ylabel("Sum Data Rate (bits/s)")
plt.legend(["TD3", "DDPG"], loc="upper right")

#plt.xlabel("Timestep(t)")
#plt.ylabel("Fairness Index")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users", "11 Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
