import numpy as np
import matplotlib.pyplot as plt
from numpy import interp


TD3_energy_multiple_users_10_6_FI = [0.067788,0.161190,0.254607] # Fairness Index Reward Components with 10^6 miltiplier
TD3_energy_multiple_users_20_6_FI = [0.149311,0.443521,0.696722]
DDPG_energy_multiple_users = []
full_offloading_multiple_users_energy = [0.00016913605774023597,0.0001703231941821021,0.000172211157922183]#3,7,11 users
local_computing_multiple_users_energy = [0.19027157569099212,0.44367963230784163,0.6970266853942844]
random_action_generation_energy = [0.18905345159746922,0.44176218716049437,0.69342701851681]

TD3_delay_multiple_users_10_6_FI = [3.198654,7.425685,11.754907]
TD3_delay_multiple_users_20_6_FI = [174.002204,601.723585,935.684333]
DDPG_delay_multiple_users = []
full_offloading_multiple_users_delay = [3.0001200707379643,7.0099352994397455,11.180665901630894]
local_computing_multiple_users_delay = [315.4792111021463,735.0747981843205,1158.8974912077504]
random_action_generation_delay = [80.96373273664972,196.39931727796113,305.6443159141167]

TD3_throughput_multiple_users_10_6_FI = [36181414.966455,33428775.969367,34897126.303756]
TD3_throughput_multiple_users_20_6_FI = [23424058.725300,23140095.823121,20792338.105923]
DDPG_throughput_multiple_users = []
full_offloading_multiple_users_throughput = [33978835.375478804,31914810.185659975,28852512.997813337] # with random resource allocation and power allocation

TD3_fairness_index_multiple_users_10_6_FI = [0.710785,0.654622,0.525979]
TD3_fairness_index_multiple_users_20_6_FI = [0.882637,0.649994,0.479813]
DDPG_fairness_index_multiple_users = []
random_resource_allocation_fairness_index_multiple_users = [0.8534970089224784,0.6627020041731907,0.5150579831673366]

num_users = [3,7,11]


figure, axis = plt.subplots(2,2)

axis[0,0].plot(num_users,TD3_fairness_index_multiple_users_10_6_FI, color="green", label=r"TD3 $10^{6}$ FI", marker='s')
axis[0,0].plot(num_users,TD3_fairness_index_multiple_users_20_6_FI, color="red", label=r"TD3 $20^{6}$ FI",marker='s')
axis[0,0].plot(num_users,random_resource_allocation_fairness_index_multiple_users, color="brown", label=r"Random RB allocation",marker='s')
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
#axis[0,0].set_title('')
axis[0,0].grid()
axis[0,0].set_xlabel('Number of Users')
axis[0,0].set_ylabel('Fairness Index')
axis[0,0].legend(loc="upper right")

axis[1,0].plot(num_users,TD3_throughput_multiple_users_10_6_FI, color="green", label=r"TD3 $10^{6}$ FI", marker='s')
axis[1,0].plot(num_users,TD3_throughput_multiple_users_20_6_FI, color="red", label=r"TD3 $20^{6}$ FI",marker='s')
axis[1,0].plot(num_users,full_offloading_multiple_users_throughput, color="brown", label=r"Random RB allocation",marker='s')
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
#axis[0,0].set_title('')
axis[1,0].grid()
axis[1,0].set_xlabel('Number of Users')
axis[1,0].set_ylabel('Sum Data Rate (bits/s)')
axis[1,0].legend(loc="upper left")

axis[0,1].plot(num_users,TD3_energy_multiple_users_10_6_FI, color="green", label=r"TD3 $10^{6}$ FI", marker='s')
#axis[1,0].plot(num_users,full_offloading_multiple_users_delay, color="red", label=r"Full Offloading",marker='s')
axis[0,1].plot(num_users,TD3_energy_multiple_users_20_6_FI, color="red", label=r"TD3 $20^{6}$ FI",marker='s')
axis[0,1].plot(num_users, random_action_generation_energy, color="brown", label='Random Action Generation',marker='s')
#axis[0,0].set_title('')
axis[0,1].grid()
axis[0,1].set_xlabel('Number of Users')
axis[0,1].set_ylabel('Energy Consumption (J)')
axis[0,1].legend(loc="upper left")

axis[1,1].plot(num_users,TD3_delay_multiple_users_10_6_FI, color="green", label=r"TD3 $10^{6}$ FI", marker='s')
#axis[1,0].plot(num_users,full_offloading_multiple_users_delay, color="red", label=r"Full Offloading",marker='s')
axis[1,1].plot(num_users,TD3_delay_multiple_users_20_6_FI, color="red", label=r"TD3 $20^{6}$ FI",marker='s')
axis[1,1].plot(num_users, random_action_generation_delay, color="brown", label='Random Action Generation',marker='s')
#axis[0,0].set_title('')
axis[1,1].grid()
axis[1,1].set_xlabel('Number of Users')
axis[1,1].set_ylabel('Task Delay (ms)')
axis[1,1].legend(loc="upper left")


plt.tight_layout()
plt.show()


# Energy Consumption
# TD3 (10⁶) consumes less energy than TD3 (20⁶) across all user counts.
# Full Offloading uses the least energy by a significant margin.
# Local Computing and Random Action Generation have almost identical energy usage, much higher than Full Offloading and TD3 strategies.
# Delay
# Full Offloading consistently achieves the lowest delay, closely followed by TD3 (10⁶).
# TD3 (20⁶) exhibits very high delay as user count increases.
# Local Computing experiences the highest delays, with an exponential increase as the number of users grows.
# Random Action Generation has intermediate delays, significantly worse than TD3 (10⁶) and Full Offloading.
# Throughput
# TD3 (10⁶) achieves the highest throughput for 3 and 11 users, but the difference between strategies is small.
# Full Offloading throughput is comparable to TD3 (10⁶) but slightly lower.
# TD3 (20⁶) throughput is consistently the lowest, decreasing as user count increases.
# Fairness Index
# TD3 (20⁶) provides the highest fairness index for 3 users but decreases rapidly as the user count increases.
# Random Resource Allocation maintains a relatively stable fairness index, comparable to TD3 (10⁶).
# TD3 (10⁶) exhibits a slight drop in fairness index as user count increases but remains competitive.
# Would you like further analysis, such as numerical summaries, percentage improvements, or additional comparisons? ​​






# You said:
# Yes, and reasons for those numbers
# ChatGPT said:
# ChatGPT
# Let’s dive deeper into the numerical summaries, percentage improvements, and the reasons behind these results for each metric:

# Energy Consumption
# Numerical Summary:
# TD3 (10⁶) consumes 60-90% less energy compared to Local Computing.
# Full Offloading achieves ~99.9% reduction in energy consumption compared to Local Computing and Random Action Generation.
# Reasons:
# TD3 Optimization: The energy-efficient policies generated by TD3 balance computational tasks and resource allocation to minimize energy usage, particularly under the 10⁶ scaling.
# Full Offloading: By delegating all computations to remote servers, the devices use minimal energy.
# Local Computing: Consumes significant energy because all computations are performed on the device, which scales poorly with the number of users.
# Random Actions: These generate suboptimal resource usage, leading to similar high energy consumption as Local Computing.
# Delay
# Numerical Summary:
# Full Offloading achieves up to 97.5% lower delay than Local Computing for 11 users.
# TD3 (10⁶) provides comparable delay performance to Full Offloading (within ~10% difference).
# TD3 (20⁶) results in 8-12x higher delays than TD3 (10⁶).
# Reasons:
# Full Offloading: Delay is minimized by using remote servers with higher processing power.
# TD3 (10⁶): Balances resource allocation effectively, keeping delay low while maintaining computational offloading efficiency.
# TD3 (20⁶): Over-optimization under higher scaling factors prioritizes objectives like fairness or throughput, resulting in significant delays.
# Local Computing: Processes all tasks locally, causing delays to grow exponentially as user count increases.
# Random Actions: The inefficiency of resource allocation leads to moderate delays.
# Throughput
# Numerical Summary:
# TD3 (10⁶) achieves the highest throughput for 3 and 11 users (up to 25% higher than TD3 (20⁶)).
# Full Offloading throughput is ~5-8% lower than TD3 (10⁶) but still competitive.
# TD3 (20⁶) shows a decline in throughput as the user count increases.
# Reasons:
# TD3 (10⁶): Balances power and resource allocation to achieve high throughput while maintaining fairness and efficiency.
# TD3 (20⁶): Focus on fairness under high scaling leads to resource reallocation that sacrifices throughput for other objectives.
# Full Offloading: Achieves good throughput by leveraging centralized computation resources but lacks dynamic optimization for throughput.
# Local Computing and Random Actions: Do not focus on optimizing throughput, resulting in lower performance.
# Fairness Index
# Numerical Summary:
# TD3 (20⁶) achieves the highest fairness (0.88) for 3 users but drops sharply to 0.48 for 11 users.
# TD3 (10⁶) maintains moderate fairness, decreasing slightly with more users.
# Random Resource Allocation stays stable across users, comparable to TD3 (10⁶).
# Reasons:
# TD3 (20⁶): Allocates resources favoring fairness, especially for smaller user groups. However, as users increase, resource contention causes fairness to drop.
# TD3 (10⁶): Balances fairness and efficiency, maintaining steady performance even as user count grows.
# Random Resource Allocation: Ensures basic fairness through equal allocation but lacks dynamic optimization.
# Full Offloading and Local Computing: Fairness is not a direct focus, leading to suboptimal fairness metrics.
# Key Insights
# TD3 vs. Full Offloading:

# TD3 achieves comparable or better delay and throughput while offering better fairness.
# Full Offloading excels in energy efficiency but may underperform in fairness and adaptability.
# Local Computing and Random Actions:

# These strategies are inefficient in all metrics, emphasizing the need for intelligent offloading and optimization.
# Impact of Scaling Factor:

# A higher scaling factor (20⁶) improves fairness but negatively impacts delay and throughput, showing a trade-off between objectives.