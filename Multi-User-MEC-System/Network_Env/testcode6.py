import numpy as np
#10mbits/sec
#1 J
snr = 16384
p_values = []

for x in range(0,1000000000):
    h = np.random.exponential(1)
    p = snr/h
    p_values.append(p)

#print(p_values)
print('max power: ', max(p_values), 'min power: ', min(p_values))