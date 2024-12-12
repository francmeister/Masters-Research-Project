import matplotlib.pyplot as plt
import numpy as np
# Fix number of urllc users to 8
number_of_embb_users = [3,7,11]
throughput_values_with_z_known = [30154261.193841,22354427.346308,24160555.946325]
throughput_values_with_z_unknown = [26233332.318159,21303384.199325,23342317.294619] # assume z is max (set p=1)

# Separate bar plots for each metric
fig, axes = plt.subplots(1, 1)

# Data Rate Plot
axes.plot(number_of_embb_users, throughput_values_with_z_known, color='green', label='z known', marker='o')
axes.plot(number_of_embb_users, throughput_values_with_z_unknown, color='brown', label='z unknown, assume p=1', marker='o')
axes.set_ylabel('Sum Data Rate (bps)')
axes.set_xlabel('Number of eMBB users')
axes.legend()
axes.grid()

plt.show()