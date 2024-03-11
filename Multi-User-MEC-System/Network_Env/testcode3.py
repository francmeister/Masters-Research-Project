import numpy as np
import matplotlib.pyplot as plt

limit = 10
sequence = list(range(1, limit + 1))

random_number = np.random.randint(0, len(sequence), 1)

print(random_number)
