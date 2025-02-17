import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 20  # Number of trials
p = 0.7  # Probability of success

# X values (possible outcomes)
# x = np.arange(0, n+1)
x = np.arange(0, 1, 0.1)

print(x)

# Compute CDF
cdf = binom.cdf(x, n, p)
cdf = binom.ppf(x, n, p)

# Plot CDF
plt.step(x, cdf, where="post", label=f'Binomial CDF (n={n}, p={p})')
plt.xlabel('Number of successes')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function of Binomial Distribution')
plt.grid()
plt.legend()
plt.show()
