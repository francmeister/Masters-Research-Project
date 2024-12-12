import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

urllc_reliability_reward_3_embb_users = np.load('urllc_reliability_reward_3_embb_users.npy')
urllc_reliability_reward_7_embb_users = np.load('urllc_reliability_reward_7_embb_users.npy')
urllc_reliability_reward_11_embb_users = np.load('urllc_reliability_reward_11_embb_users.npy')

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

urllc_reliability_reward_3_embb_users_smooth = moving_average(urllc_reliability_reward_3_embb_users, window_size)
urllc_reliability_reward_7_embb_users_smooth = moving_average(urllc_reliability_reward_7_embb_users, window_size)
urllc_reliability_reward_11_embb_users_smooth = moving_average(urllc_reliability_reward_11_embb_users, window_size)


number_of_embb_users = [3,7,11]
#number of URLLC users = 8
# #Models trained with p=0.1
# outage_probabilities_0_1 = [0.108051,0.083359,0.115015]
# outage_probabilities_0_5 = [0.753837,0.697037,0.617304]
# outage_probabilities_0_9 = [0.985717,0.983289,0.976260]

#number of URLLC users = 8
#Models trained with p=0.5
outage_probabilities_0_1 = [0.001254,0.005182,0.018992]
outage_probabilities_0_5 = [0.112898,0.146539,0.245808] #0.093959 (3 users)
outage_probabilities_0_9 = [0.480960,0.553509,0.659966]

figure, axis = plt.subplots(2,2)

axis[0,0].plot(timesteps[window_size-1:], urllc_reliability_reward_3_embb_users_smooth, color="green", label=r"3 eMBB Users")
axis[0,0].plot(timesteps[window_size-1:], urllc_reliability_reward_7_embb_users_smooth, color="red", label=r"7 eMBB Users")
axis[0,0].plot(timesteps[window_size-1:], urllc_reliability_reward_11_embb_users_smooth, color="brown", label=r"11 eMBB Users")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,0].set_title('Reliability Constraint Rewards')
axis[0,0].grid()
axis[0,0].set_xlabel('Timestep')
axis[0,0].legend(loc="lower right")

axis[0,1].plot(timesteps[window_size-1:], urllc_total_rate_3_users_smooth, color="green", label=r"3 eMBB Users")
axis[0,1].plot(timesteps[window_size-1:], urllc_total_rate_7_users_smooth, color="red", label=r"7 eMBB Users")
axis[0,1].plot(timesteps[window_size-1:], urllc_total_rate_11_users_smooth, color="brown", label=r"11 eMBB Users")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[0,1].set_title('URLLC Users Sum Data Rate')
axis[0,1].grid()
axis[0,1].set_xlabel('Timestep')
axis[0,1].set_ylabel('Sum Data Rate (bits/slot)')
axis[0,1].legend(loc="lower right")

axis[1,0].plot(number_of_embb_users, outage_probabilities_0_1, color="green", label=r"p=0.1", marker='s')
axis[1,0].plot(number_of_embb_users, outage_probabilities_0_5, color="red", label=r"p=0.5", marker='s')
axis[1,0].plot(number_of_embb_users, outage_probabilities_0_9, color="brown", label=r"p=0.9", marker='s')
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[1,0].set_title('Outage Probabilities Inference')
axis[1,0].grid()
axis[1,0].set_xlabel('Number of eMBB users')
axis[1,0].set_ylabel('Outage Probability Value')
axis[1,0].legend(loc="lower right")

# axis[1,0].plot(timesteps[window_size-1:], F_L_inverse_3_embb_users_smooth, color="green", label=r"3 eMBB Users")
# axis[1,0].plot(timesteps[window_size-1:], F_L_inverse_7_embb_users_smooth, color="red", label=r"7 eMBB Users")
# axis[1,0].plot(timesteps[window_size-1:], F_L_inverse_11_embb_users_smooth, color="brown", label=r"11 eMBB Users")
# #axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
# axis[1,0].set_title('F_L_inverse')
# axis[1,0].grid()
# axis[1,0].set_xlabel('Timestep')
# axis[1,0].set_ylabel('F_L_inverse Value')
# axis[1,0].legend(loc="lower right")



axis[1,1].plot(timesteps[window_size-1:], outage_probabilties_3_embb_users_smooth, color="green", label=r"3 eMBB Users")
axis[1,1].plot(timesteps[window_size-1:], outage_probabilties_7_embb_users_smooth, color="red", label=r"7 eMBB Users")
axis[1,1].plot(timesteps[window_size-1:], outage_probabilties_11_embb_users_smooth, color="brown", label=r"11 eMBB Users")
#axis[0,0].plot(timesteps_256_steps[window_size-1:], overall_users_reward_256_steps_smooth, color="blue", label='3 Users')
axis[1,1].set_title('Outage Probability')
axis[1,1].grid()
axis[1,1].set_xlabel('Timestep')
axis[1,1].set_ylabel('Outage Probability Value')
axis[1,1].legend(loc="lower right")


plt.tight_layout()
plt.show()