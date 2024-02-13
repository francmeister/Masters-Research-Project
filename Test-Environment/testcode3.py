import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,1000,1)
total_gains = []

for y in x:
    small_scale_channel_gain = np.random.rayleigh(1)
    large_scale_channel_gain = np.random.lognormal(0.0,1.0)
    total_gain = large_scale_channel_gain*small_scale_channel_gain#*self.large_scale_channel_gain*self.pathloss_gain
    total_gains.append(total_gain)

print('Max channel gain: ', np.max(total_gains), ' Min channel gain: ', np.min(total_gain))
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.boxplot(total_gains)
#plt.plot(allocated_RBs,local_energy,color = "blue")
#plt.plot(allocated_RBs,transmit_energy,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
plt.legend(["total gain"])
plt.xlabel("Transmit Power (dBm)")
plt.ylabel("Throughput (bits/s)")
plt.title("Throughput vs Transmit Power, Allocated RBs = 15, offloading ratio = 0.5")
plt.show()
