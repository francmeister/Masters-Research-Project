import numpy as np
import matplotlib.pyplot as plt

global_reward = np.load('global_reward.npy')
timesteps = np.load('TD3_NetworkEnv-v0_0.npy')

print(global_reward)

timesteps = len(global_reward)


plt.plot(timesteps, global_reward, color ="blue")

plt.show()

