import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


p = [0.2,0.4,0.6,0.8]
op_2_embb_users = [0.000673,0.001495,0.002223, 0.004759, 0.009209]
op_1_embb_users = []
op_3_embb_users = []
op_5_embb_users = []
op_7_embb_users = []
op_9_embb_users = []
# plt.plot(p,throughputs_1_users,'--', marker='*', ms = 10,color="green")
# plt.plot(p,throughputs_3_users,'--', marker='*', ms = 10,color="blue")
# plt.plot(p,throughputs_5_users,'--', marker='*', ms = 10,color="red")


plt.xlabel("Probabilty of generating task")
plt.ylabel("Throughput")
#plt.legend(["SR=0.1, FI=0.9","SR=0.2, FI=0.8", "SR=0.4, FI=0.6", "SR=0.8, FI=0.2", "SR=1, FI=1"], loc="upper left")
#plt.grid()

plt.tight_layout()

plt.show()
