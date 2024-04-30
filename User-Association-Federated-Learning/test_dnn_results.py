
import numpy as np
import matplotlib.pyplot as plt


average_rewards_in_memory = np.load('average_rewards_in_memory.npy')

channel_rates = np.load('channel_rates.npy')

timesteps_average_rewards_in_memory = []
timesteps_channel_rates = []
timesteps_training_loss = []

x = 0
for gb in average_rewards_in_memory:
    timesteps_average_rewards_in_memory.append(x)
    x+=1

x = 0
for gb in channel_rates:
    timesteps_channel_rates.append(x)
    x+=1


#plt.plot(timesteps_average_rewards_in_memory, average_rewards_in_memory, color ="blue")
plt.plot(timesteps_channel_rates, channel_rates, color ="blue")
#plt.plot(timesteps_training_loss, training_loss, color ="blue")

plt.show()