import numpy as np
import random

max_samples = 20
num_users = 5
user_ids = []
for x in range(0,max_samples):
    user_ids.append(random.randint(1, num_users))

#Random channel conditions
channel_conditions = []
for x in range(0,max_samples):
    channel_conditions.append(random.random())

#Random queue lengths
queue_lengths = []
for x in range(0,max_samples):
    queue_lengths.append(random.random())

#Random associations
user_associations = []
for x in range(0,max_samples):
    user_associations.append(random.sample(range(8),num_users))

X_inputs = [user_ids,channel_conditions,queue_lengths]
X_inputs = np.array(X_inputs)

print(X_inputs)
