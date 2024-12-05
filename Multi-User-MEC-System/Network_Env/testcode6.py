
import numpy as np

expl_noise = 0.3
action_space_shape = 5
print(np.random.normal(0, expl_noise, size=action_space_shape))