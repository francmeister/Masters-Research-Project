import numpy as np

result = np.load('sum_allocations_per_RB_matrix.npy')
timesteps = np.load('TD3_NetworkEnv-v0_0.npy')

print(len(result))
print(len(timesteps))

print(result)