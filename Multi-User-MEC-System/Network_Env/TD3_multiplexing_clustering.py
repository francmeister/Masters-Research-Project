# import matplotlib.pyplot as plt
# import numpy as np

# number_of_embb_users = [3,7,11]

# outage_probability_10_8_urllc_reliability_reward = []
# outage_probability_15_8_urllc_reliability_reward = [,,0.245808]
# outage_probability_20_8_urllc_reliability_reward = [,0.146539,0.245808]
# outage_probability_30_8_urllc_reliability_reward = [,0.146539,0.234804]

# embb_throughput_10_8_urllc_reliability_reward = []
# embb_throughput_15_8_urllc_reliability_reward = [,,23509171.703848]
# embb_throughput_20_8_urllc_reliability_reward = [,,23509171.703848]
# embb_throughput_30_8_urllc_reliability_reward = [,22354427.346308,21493332.666826]

# embb_energy_10_8_urllc_reliability_reward = []
# embb_energy_15_8_urllc_reliability_reward = [,,0.696766]
# embb_energy_20_8_urllc_reliability_reward = [,,0.696766]
# embb_energy_30_8_urllc_reliability_reward = [,0.443501,0.696766]

# embb_delay_10_8_urllc_reliability_reward = []
# embb_delay_15_8_urllc_reliability_reward = [,,951.413534]
# embb_delay_20_8_urllc_reliability_reward = [,,951.401470]
# embb_delay_30_8_urllc_reliability_reward = [,606.964678,951.395289]

outage_probabilites = [0.112898, 0.112898,0.112898]
urllc_reward_values = [10**8, 15**8, 20**8, 30**8]

import matplotlib.pyplot as plt
import numpy as np

urllc_reliability_reward_3_embb_users_10_8 = np.load('urllc_reliability_reward_3_embb_users_10_8.npy')
urllc_reliability_reward_3_embb_users_15_8 = np.load('urllc_reliability_reward_3_embb_users_15_8.npy')
urllc_reliability_reward_3_embb_users_20_8 = np.load('urllc_reliability_reward_3_embb_users_20_8.npy')
urllc_reliability_reward_3_embb_users_30_8 = np.load('urllc_reliability_reward_3_embb_users_30_8.npy')

outage_probabilties_3_embb_users_random = np.load('outage_probabilties_3_embb_users_random.npy')
outage_probabilties_3_embb_users_distance = np.load('outage_probabilties_3_embb_users_distance.npy')
outage_probabilties_7_embb_users_random = np.load('outage_probabilties_7_embb_users_random.npy')
outage_probabilties_7_embb_users_distance = np.load('outage_probabilties_7_embb_users_distance.npy')
outage_probabilties_11_embb_users_random = np.load('outage_probabilties_11_embb_users_random.npy')
outage_probabilties_11_embb_users_distance = np.load('outage_probabilties_11_embb_users_distance.npy')


timestep_rewards_energy_throughput_3_embb_users_10_0 = np.load('timestep_rewards_energy_throughput_3_embb_users_10_0.npy')
timestep_rewards_energy_throughput_3_embb_users_10_8 = np.load('timestep_rewards_energy_throughput_3_embb_users_10_8.npy')
timestep_rewards_energy_throughput_3_embb_users_15_8 = np.load('timestep_rewards_energy_throughput_3_embb_users_15_8.npy')
timestep_rewards_energy_throughput_3_embb_users_20_8 = np.load('timestep_rewards_energy_throughput_3_embb_users_20_8.npy')
timestep_rewards_energy_throughput_3_embb_users_30_8 = np.load('timestep_rewards_energy_throughput_3_embb_users_30_8.npy')

throughputs_3_embb_users_10_0 = timestep_rewards_energy_throughput_3_embb_users_10_0[:,3]
throughputs_3_embb_users_10_8 = timestep_rewards_energy_throughput_3_embb_users_10_8[:,3]
throughputs_3_embb_users_15_8 = timestep_rewards_energy_throughput_3_embb_users_15_8[:,3]
throughputs_3_embb_users_20_8 = timestep_rewards_energy_throughput_3_embb_users_20_8[:,3]
throughputs_3_embb_users_30_8 = timestep_rewards_energy_throughput_3_embb_users_30_8[:,3]

timesteps_3_embb_users_10_0 = timestep_rewards_energy_throughput_3_embb_users_10_0[:,0]

F_L_inverse_3_embb_users = np.load('F_L_inverse_3_embb_users.npy')
F_L_inverse_7_embb_users = np.load('F_L_inverse_3_embb_users.npy')
F_L_inverse_11_embb_users = np.load('F_L_inverse_3_embb_users.npy')

urllc_total_rate_3_users = np.load('urllc_total_rate_3_embb_users.npy')
urllc_total_rate_7_users = np.load('urllc_total_rate_7_embb_users.npy')
urllc_total_rate_11_users = np.load('urllc_total_rate_11_embb_users.npy')

outage_probabilties_3_embb_users = np.load('outage_probabilties_3_embb_users.npy')
outage_probabilties_7_embb_users = np.load('outage_probabilties_7_embb_users.npy')
outage_probabilties_11_embb_users = np.load('outage_probabilties_11_embb_users.npy')

rewards_throughput_energy = np.load('timestep_rewards_energy_throughput.npy')
timesteps = rewards_throughput_energy[:,0]

def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100

F_L_inverse_3_embb_users_smooth = moving_average(F_L_inverse_3_embb_users, window_size)
F_L_inverse_7_embb_users_smooth = moving_average(F_L_inverse_7_embb_users, window_size)
F_L_inverse_11_embb_users_smooth = moving_average(F_L_inverse_11_embb_users, window_size)

urllc_total_rate_3_users_smooth = moving_average(urllc_total_rate_3_users, window_size)
urllc_total_rate_7_users_smooth = moving_average(urllc_total_rate_7_users, window_size)
urllc_total_rate_11_users_smooth = moving_average(urllc_total_rate_11_users, window_size)

outage_probabilties_3_embb_users_smooth = moving_average(outage_probabilties_3_embb_users, window_size)
outage_probabilties_7_embb_users_smooth = moving_average(outage_probabilties_7_embb_users, window_size)
outage_probabilties_11_embb_users_smooth = moving_average(outage_probabilties_11_embb_users, window_size)

urllc_reliability_reward_3_embb_users_10_8 = moving_average(urllc_reliability_reward_3_embb_users_10_8, window_size)
urllc_reliability_reward_3_embb_users_15_8 = moving_average(urllc_reliability_reward_3_embb_users_15_8, window_size)
urllc_reliability_reward_3_embb_users_20_8 = moving_average(urllc_reliability_reward_3_embb_users_20_8, window_size)
urllc_reliability_reward_3_embb_users_30_8 = moving_average(urllc_reliability_reward_3_embb_users_30_8, window_size)

outage_probabilties_3_embb_users_random = moving_average(outage_probabilties_3_embb_users_random, window_size)
outage_probabilties_3_embb_users_distance = moving_average(outage_probabilties_3_embb_users_distance, window_size)
outage_probabilties_7_embb_users_random = moving_average(outage_probabilties_7_embb_users_random, window_size)
outage_probabilties_7_embb_users_distance = moving_average(outage_probabilties_7_embb_users_distance, window_size)
outage_probabilties_11_embb_users_random = moving_average(outage_probabilties_11_embb_users_random, window_size)
outage_probabilties_11_embb_users_distance = moving_average(outage_probabilties_11_embb_users_distance, window_size)

#outage_probabilties_3_embb_users_random = moving_average(outage_probabilties_3_embb_users_20_8, window_size)
#outage_probabilties_3_embboutage_probabilties_3_embb_users_random_users_30_8 = moving_average(outage_probabilties_3_embb_users_30_8, window_size)
# urllc_reliability_reward_7_embb_users_smooth = moving_average(urllc_reliability_reward_7_embb_users, window_size)
# urllc_reliability_reward_11_embb_users_smooth = moving_average(urllc_reliability_reward_11_embb_users, window_size)
throughputs_3_embb_users_10_0 =  moving_average(throughputs_3_embb_users_10_0, window_size)
throughputs_3_embb_users_10_8 =  moving_average(throughputs_3_embb_users_10_8, window_size)
throughputs_3_embb_users_15_8 =  moving_average(throughputs_3_embb_users_15_8, window_size)
throughputs_3_embb_users_20_8 =  moving_average(throughputs_3_embb_users_20_8, window_size)
throughputs_3_embb_users_30_8 =  moving_average(throughputs_3_embb_users_30_8, window_size)

number_of_embb_users = [3,7,11]
#number of URLLC users = 8
# #Models trained with p=0.1
# outage_probabilities_0_1 = [0.108051,0.083359,0.115015]
# outage_probabilities_0_5 = [0.753837,0.697037,0.617304]
# outage_probabilities_0_9 = [0.985717,0.983289,0.976260]

#number of URLLC users = 8
#Models trained with p=0.5
outage_probabilities_random = [0.008779,0.113049,0.181745]
outage_probabilities_distance = [0.112898,0.146539,0.245808] #0.093959 (3 users)

figure, axis = plt.subplots(2,2)

axis[0,0].plot(timesteps[window_size-1:], outage_probabilties_3_embb_users_random, color="green", label=r"Random Clustering")
axis[0,0].plot(timesteps[window_size-1:], outage_probabilties_3_embb_users_distance, color="red", label=r"Near Distance Clustering")
# axis[0,0].plot(timesteps[window_size-1:], urllc_reliability_reward_3_embb_users_20_8, color="brown", label=r"$20^{8}$ reliability reward")
# axis[0,0].plot(timesteps[window_size-1:], urllc_reliability_reward_3_embb_users_30_8, color="blue", label=r"$30^{8}$ reliability reward")
# #axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,0].set_title('3 Users Outage Probability Training')
axis[0,0].grid()
axis[0,0].set_xlabel('Timestep')
axis[0,0].legend(loc="upper right")

axis[0,1].plot(timesteps[window_size-1:], outage_probabilties_7_embb_users_random, color="green", label=r"Random Clustering")
axis[0,1].plot(timesteps[window_size-1:], outage_probabilties_7_embb_users_distance, color="red", label=r"Near Distance Clustering")
# axis[0,1].plot(timesteps[window_size-1:], outage_probabilties_3_embb_users_20_8, color="brown", label=r"$20^{8}$ reliability reward")
# axis[0,1].plot(timesteps[window_size-1:], outage_probabilties_3_embb_users_30_8, color="blue", label=r"$30^{8}$ reliability reward")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,1].set_title('7 Users Outage Probabilities Training')
axis[0,1].grid()
axis[0,1].set_xlabel('Timestep')
# axis[0,1].set_ylabel('Sum Data Rate (bits/slot)')
axis[0,1].legend(loc="upper right")

axis[1,0].plot(timesteps[window_size-1:], outage_probabilties_11_embb_users_random, color="green", label=r"Random Clustering")
axis[1,0].plot(timesteps[window_size-1:], outage_probabilties_11_embb_users_distance, color="red", label=r"Near Distance Clustering")
# axis[1,0].plot(timesteps[window_size-1:], throughputs_3_embb_users_15_8, color="red", label=r"$15^{8}$ reliability reward")
# axis[1,0].plot(timesteps[window_size-1:], throughputs_3_embb_users_20_8, color="brown", label=r"$20^{8}$ reliability reward")
# axis[1,0].plot(timesteps[window_size-1:], throughputs_3_embb_users_30_8, color="blue", label=r"$30^{8}$ reliability reward")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[1,0].set_title('11 Users Outage Probability Training')
axis[1,0].grid()
axis[1,0].set_xlabel('Timestep')
#axis[1,0].set_ylabel('Data Rate (bps)')
axis[1,0].legend(loc="upper right")

axis[1,1].plot(number_of_embb_users, outage_probabilities_random, color="green", label=r"Random Clustering", marker='o')
axis[1,1].plot(number_of_embb_users, outage_probabilities_distance, color="red", label=r"Near Distance Clustering", marker='o')
#axis[1,0].plot(timesteps[window_size-1:], F_L_inverse_11_embb_users_smooth, color="brown", label=r"11 eMBB Users")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[1,1].set_title('Outage Probabilities Inference')
axis[1,1].grid()
axis[1,1].set_xlabel('Number of Users')
axis[1,1].set_ylabel('Outage Probability')
axis[1,1].legend(loc="upper left")



# axis[1,1].plot(timesteps[window_size-1:], outage_probabilties_3_embb_users_smooth, color="green", label=r"3 eMBB Users")
# axis[1,1].plot(timesteps[window_size-1:], outage_probabilties_7_embb_users_smooth, color="red", label=r"7 eMBB Users")
# axis[1,1].plot(timesteps[window_size-1:], outage_probabilties_11_embb_users_smooth, color="brown", label=r"11 eMBB Users")
# #axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
# axis[1,1].set_title('Outage Probability')
# axis[1,1].grid()
# axis[1,1].set_xlabel('Timestep')
# axis[1,1].set_ylabel('Outage Probability Value')
# axis[1,1].legend(loc="lower right")


plt.tight_layout()
plt.show()

