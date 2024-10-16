import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

# A total of 12 urllc users
one_embb_user = [0.000006,0.003152,0.085441, 0.550993]
three_embb_user = [0.000004,0.002257, 0.068737,0.492101]
five_embb_user = []
seven_embb_user = [0.000005,0.002661,0.077574, 0.531721]
nine_embb_user = [0.000018,0.005686, 0.115723, 0.607640]
prob_gen_task = [0.2,0.4,0.6,0.8]

plt.plot(prob_gen_task, one_embb_user, color='black', marker='*')
plt.plot(prob_gen_task, three_embb_user, color='green', marker='*')
plt.plot(prob_gen_task, seven_embb_user, color='blue', marker='*')
plt.plot(prob_gen_task, nine_embb_user, color='red', marker='*')


plt.xlabel("URLLC Users' Probability of Generating a Task")
plt.ylabel("URLLC Users Outage Probability ($\phi$)")
plt.legend(["1 eMBB User", "3 eMBB Users", "7 eMBB Users", "9 eMBB Users"], loc="upper left")

#plt.xlabel("Timestep(t)")
#plt.ylabel("Fairness Index")
#plt.legend(["1 User","3 Users", "5 Users", "7 Users", "9 Users", "11 Users"], loc="upper left")
plt.grid()

plt.tight_layout()

plt.show()
