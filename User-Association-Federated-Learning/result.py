import numpy as np
import matplotlib.pyplot as plt

global_reward = np.load('average_reward_in_memory.npy')
#timesteps = np.load('TD3_NetworkEnv-v0_0.npy')

print(len(global_reward))

timesteps = []#len(global_reward)

x = 0
for gb in global_reward:
    timesteps.append(x)
    x+=1



plt.plot(timesteps, global_reward, color ="blue")

plt.show()

