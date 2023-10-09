import math
import numpy as np
import random
from numpy import interp
import pandas as pd

system_bandwidth_Hz = 120*math.pow(10,6)
subcarrier_bandwidth_Hz = 15*math.pow(10,3) # 15kHz
num_subcarriers_per_RB = 12
RB_bandwidth_Hz = subcarrier_bandwidth_Hz*num_subcarriers_per_RB
num_RB = int(system_bandwidth_Hz/RB_bandwidth_Hz)
long_TTI = 1 #1ms

noise_spectral_density_dbm = -174 # -174dBM/Hz
noise_spectral_density_W = (math.pow(10,(noise_spectral_density_dbm/10)))/1000

assigned_transmit_power_dBm = 20
assigned_transmit_power_W = 0
assigned_transmit_power_W = (math.pow(10,(assigned_transmit_power_dBm/10)))/1000

allocated_RBs = []

number_of_allocated_RBs = 1
total_gains = []
achieved_rates = []
def calculate_channel_gain():
    small_scale_channel_gain = np.random.rayleigh(1)
    total_gain = small_scale_channel_gain#*self.large_scale_channel_gain*self.pathloss_gain
    return total_gain

def calculate_channel_rate(total_gain):
        RB_bandwidth = RB_bandwidth_Hz
        noise_spectral_density = noise_spectral_density_W
        channel_rate_numerator = assigned_transmit_power_W * total_gain
        channel_rate_denominator = noise_spectral_density*RB_bandwidth
        channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
        return channel_rate

def transmit_to_SBS(total_gain):
        #Calculate the bandwidth achieved on each RB
        achieved_RB_channel_rates = []
        #print('number of allocated RBs: ', len(self.allocate(d_RBs))
        number_of_allocated_RBs = 5
        allocated_RBs.clear()
        for i in range(1,number_of_allocated_RBs):
            allocated_RBs.append(i)

        for RB in allocated_RBs:
            achieved_RB_channel_rate = calculate_channel_rate(total_gain)
            achieved_RB_channel_rates.append(achieved_RB_channel_rate)

        print('achieved_RB_channel_rates matrix: ', achieved_RB_channel_rates)
        achieved_channel_rate = sum(achieved_RB_channel_rates)
        print('total gain: ', total_gain, " achieved channel rate: ", achieved_channel_rate )
        return achieved_channel_rate

number_of_timesteps = 10

for i in range(1,number_of_timesteps):
    total_gain = calculate_channel_gain()
    total_gains.append(total_gain)
    channel_rate = transmit_to_SBS(total_gain)
    achieved_rates.append(channel_rate)
    print(' ')
    print(' ')

gains_throughputs = {
    'gains': total_gains,
    'throughputs': achieved_rates
}

df = pd.DataFrame(data=gains_throughputs)
print(df)

corr = df.corr(method='pearson')
print(corr)