import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
import math

global_reward = np.load('global_reward.npy')
average_reward_in_memory_AP_1 = np.load('average_reward_in_memory_AP_1.npy')
average_reward_in_memory_AP_2 = np.load('average_reward_in_memory_AP_2.npy')
average_reward_in_memory_AP_3 = np.load('average_reward_in_memory_AP_3.npy')
DNN_training_loss_AP_1 = np.load('DNN_training_loss_AP_1.npy')
DNN_training_loss_AP_2 = np.load('DNN_training_loss_AP_2.npy')
DNN_training_loss_AP_3 = np.load('DNN_training_loss_AP_3.npy')
print('len(DNN_training_loss_AP_1): ', len(DNN_training_loss_AP_1))
timesteps_global_reward = []
x = 0
for gb in global_reward:
    timesteps_global_reward.append(x)
    x+=1

timesteps_average_reward_in_memory_AP_1 = []
x = 0
for gb in average_reward_in_memory_AP_1:
    timesteps_average_reward_in_memory_AP_1.append(x)
    x+=1

timesteps_average_reward_in_memory_AP_2 = []
x = 0
for gb in average_reward_in_memory_AP_2:
    timesteps_average_reward_in_memory_AP_2.append(x)
    x+=1

timesteps_average_reward_in_memory_AP_3 = []
x = 0
for gb in average_reward_in_memory_AP_3:
    timesteps_average_reward_in_memory_AP_3.append(x)
    x+=1

timestep_DNN_training_loss_AP_1 = []
x = 0
for gb in DNN_training_loss_AP_1:
    timestep_DNN_training_loss_AP_1.append(x)
    x+=1

print('len(timestep_DNN_training_loss_AP_1): ', len(timestep_DNN_training_loss_AP_1))
timestep_DNN_training_loss_AP_2 = []
x = 0
for gb in DNN_training_loss_AP_2:
    timestep_DNN_training_loss_AP_2.append(x)
    x+=1

timestep_DNN_training_loss_AP_3 = []
x = 0
for gb in DNN_training_loss_AP_3:
    timestep_DNN_training_loss_AP_3.append(x)
    x+=1


def moving_average(data, window_size):
    """Compute the moving average of data."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

window_size = 100
window_size_training_DNNs = 100

global_reward_smooth = moving_average(global_reward, window_size)
DNN_training_loss_AP_1_smooth = moving_average(DNN_training_loss_AP_1, window_size_training_DNNs)
DNN_training_loss_AP_2_smooth = moving_average(DNN_training_loss_AP_2, window_size_training_DNNs)
DNN_training_loss_AP_3_smooth = moving_average(DNN_training_loss_AP_3, window_size_training_DNNs)

figure, axis = plt.subplots(3,3)


axis[0,0].plot(timesteps_global_reward[window_size-1:], global_reward_smooth)
#axis[0,0].plot(timesteps, overall_users_reward)
axis[0,0].set_title('Sum Utility Value all APs')
axis[0,0].set_xlabel('Timestep')
axis[0,0].set_ylabel('Sum Utility Value (bits/s)')
axis[0,0].grid()

axis[1,0].plot(timesteps_average_reward_in_memory_AP_1, average_reward_in_memory_AP_1)
#axis[1,0].plot(timesteps, energies)
axis[1,0].set_title('Average Utility Value in Training Memory AP 1')
axis[1,0].set_xlabel('Epoch')
axis[1,0].set_ylabel('Average Utility Value (bits/s)')
axis[1,0].grid()

axis[2,0].plot(timesteps_average_reward_in_memory_AP_2, average_reward_in_memory_AP_2)
#axis[1,0].plot(timesteps, energies)
axis[2,0].set_title('Average Utility Value in Training Memory AP 2')
axis[2,0].set_xlabel('Epoch')
axis[2,0].set_ylabel('Average Utility Value (bits/s)')
axis[2,0].grid()

axis[0,1].plot(timesteps_average_reward_in_memory_AP_3, average_reward_in_memory_AP_3)
#axis[1,0].plot(timesteps, energies)
axis[0,1].set_title('Average Utility Value in Training Memory AP 3')
axis[0,1].set_xlabel('Epoch')
axis[0,1].set_ylabel('Average Utility Value (bits/s)')
axis[0,1].grid()

axis[1,1].plot(timestep_DNN_training_loss_AP_1[window_size-1:], DNN_training_loss_AP_1_smooth)
#axis[0,1].plot(timesteps, throughputs)
axis[1,1].set_title('DNN Training Loss AP 1')
axis[1,1].set_xlabel('Training step')
axis[1,1].set_ylabel('Training Loss')
axis[1,1].grid()

axis[2,1].plot(timestep_DNN_training_loss_AP_2[window_size-1:], DNN_training_loss_AP_2_smooth)
#axis20,1].plot(timesteps, throughputs)
axis[2,1].set_title('DNN Training Loss AP 2')
axis[2,1].set_xlabel('Training step')
axis[2,1].set_ylabel('Training Loss')
axis[2,1].grid()

axis[0,2].plot(timestep_DNN_training_loss_AP_3[window_size-1:], DNN_training_loss_AP_3_smooth)
#axis[0,1].plot(timesteps, throughputs)
axis[0,2].set_title('DNN Training Loss AP 3')
axis[0,2].set_xlabel('Training step')
axis[0,2].set_ylabel('Training Loss')
axis[0,2].grid()


# axis[0].plot(evaluation_timesteps1, energy_efficiency_rewards)
# axis[0].set_title('Energy Efficiency')


# axis[1].plot(evaluation_timesteps1[window_size-1:], TD3_smooth)
# axis[1].set_title('Energy Efficiency')

#plt.plot(timesteps_TD3[window_size-1:], TD3_smooth, color="blue", label='TD3')

plt.tight_layout()

plt.show()
#plt.figure(1)
###plt.subplot(211)
#plt.plot(timesteps, fairness_index, color ="red")
#plt.subplot(212)
#plt.plot(timesteps, throughputs, color ="green")
#plt.subplot(214)
#plt.plot(timesteps, rewards, color ="blue")

#plt.legend(["reward"])
#plt.xlabel("Episodes")
#plt.ylabel("reward")
#plt.title("Evolution of Reward")


