# import numpy as np

# # Parameters
# scale = 1.0  # Scale parameter (1/lambda)
# lambda_ = 1 / scale  # Rate parameter (lambda)

# # Calculate the 99.9th percentile (practical maximum value)
# percentile = 0.999
# practical_max = -np.log(1 - percentile) / lambda_

# print("Practical maximum value (99.9th percentile):", practical_max)

# # Parameters
# scale = 1.0  # Scale parameter
# sample_size = 1000000000  # Large sample size to observe the maximum value

# # Draw samples
# samples = np.random.exponential(scale, sample_size)

# # Find the maximum value in the sample
# max_value = np.max(samples)

# print("Maximum value observed in the sample:", max_value)

import numpy as np
import matplotlib.pyplot as plt 

# Parameters
scale = 1.0  # Scale parameter
sample_size = 1000  # Sample size

# Draw samples
samples = np.random.exponential(scale, sample_size)

# Find the minimum value in the sample
min_value = np.min(samples)

print("Minimum value observed in the sample:", min_value)

# Parameters
scale = 6.0  # Scale parameter
large_sample_size = 10000  # Large sample size

# Draw samples
large_samples = np.random.exponential(scale, large_sample_size)

# Find the minimum value in the large sample
min_large_value = np.min(large_samples)

x_axis = []
for x in range(0,large_sample_size):
    x_axis.append(x)

plt.plot(x_axis, large_samples)
plt.show()


print("Minimum value observed in the large sample:", min_large_value)

