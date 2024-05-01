import numpy as np
import matplotlib.pyplot as plt

global_reward = np.load('global_reward.npy')
average_reward_in_memory = np.load('average_reward_in_memory.npy')
#timesteps = np.load('TD3_NetworkEnv-v0_0.npy')

# global_reward = global_reward + np.random.normal(0, 1000)
# x = 0
# g=0
# for gb in global_reward:
#     g+=np.random.normal(0, 200)
#     global_reward[x] = gb+g
#     x+=1



print(len(global_reward))

timesteps_global_reward = []#len(global_reward)

x = 0
for gb in global_reward:
    timesteps_global_reward.append(x)
    x+=1

timesteps = []#len(global_reward)

x = 0
for gb in average_reward_in_memory:
    timesteps.append(x)
    x+=1



plt.plot(timesteps_global_reward, global_reward, color ="blue")
#plt.plot(timesteps, average_reward_in_memory, color ="blue")

plt.show()

