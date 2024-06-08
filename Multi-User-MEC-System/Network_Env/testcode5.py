import numpy as np
import math

probs = []
for i in range(1,50):
    task_arrival_rate_tasks_per_slot = np.random.binomial(size=1,n=1,p=0.8)
    task_arrival_rate_tasks_per_slot = task_arrival_rate_tasks_per_slot[0]
    probs.append(task_arrival_rate_tasks_per_slot)

probs_sum = sum(probs)

print(probs_sum)