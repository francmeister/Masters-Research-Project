import numpy as np

f = np.random.binomial(size=1,n=1,p=0.5)
count = 1
Access_point_name = 'Access Point ' + str(count)

large_scale_gain = np.random.exponential(1,size=(1,3))

random_number = np.random.randint(0, 6, 1)
index = int(random_number)
print(index)