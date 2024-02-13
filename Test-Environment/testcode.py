import math
import random
import numpy as np
import matplotlib.pyplot as plt

packet_size_bits = 5000*8000

cpu_cycles_per_byte = 330
offload_ratio = 0.1
offload_ratios = np.arange(0,1.1,0.1)

local_energy = []
transmit_energy = []
throughput = []
total_energy = []
transmit_powers = np.arange(0,80,1)
allocated_RBs = np.arange(1,15,1)

for transmit_power in transmit_powers:
    #print(offload_ratio)
    offload_ratio = 0.5
    #transmit_power = 10
    allocated_RB = 1

    local_ratio = (1-offload_ratio)
    print('offloading ratio: ', offload_ratio)
    cycles_per_bits = cpu_cycles_per_byte*8*(local_ratio*packet_size_bits)
    energy_consumption_coefficient = math.pow(10,-15)
    cpu_clock_frequency = 5000
    #allocated_offloading_ratio = 0.5
    achieved_local_energy_consumption = energy_consumption_coefficient*math.pow(cpu_clock_frequency,2)*cycles_per_bits
    print('Energy Local: ',achieved_local_energy_consumption)
    achieved_local_processing_delay = cycles_per_bits/cpu_clock_frequency
    #print(achieved_local_processing_delay)

    distance_from_SBS = 7
    pathloss_gain = 35.3+37.6*math.log10(distance_from_SBS)#(math.pow(10,(35.3+37.6*math.log10(distance_from_SBS))))/10
    small_scale_channel_gain = np.random.rayleigh(1)
    large_scale_channel_gain = np.random.lognormal(0.0,1.0)
    total_gain = 0.1#large_scale_channel_gain*small_scale_channel_gain#*self.large_scale_channel_gain*self.pathloss_gain
    #print('pathloss gain: ',pathloss_gain)
    #print('small_scale_channel_gain: ',small_scale_channel_gain)
    #print('large scale channel gain: ', large_scale_channel_gain)
    #print('total gain: ', total_gain)

    system_bandwidth_Hz = 120*math.pow(10,6)
    subcarrier_bandwidth_Hz = 15*math.pow(10,3) # 15kHz
    num_subcarriers_per_RB = 12
    RB_bandwidth_Hz = subcarrier_bandwidth_Hz*num_subcarriers_per_RB
    num_RB = int(system_bandwidth_Hz/RB_bandwidth_Hz)
    long_TTI = 1 #1ms
    noise_spectral_density_dbm = -174 # -174dBM/Hz
    noise_spectral_density_W = (math.pow(10,(noise_spectral_density_dbm/10)))/1000
    assigned_transmit_power_dbm = transmit_power
    assigned_transmit_power_W = (math.pow(10,(assigned_transmit_power_dbm/10)))/1000
    RB_bandwidth = RB_bandwidth_Hz
    channel_rate_numerator = assigned_transmit_power_W*total_gain
    channel_rate_denominator = noise_spectral_density_W*RB_bandwidth_Hz
    channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
    #print('channel gain numerator: ', channel_rate_numerator)
    #print('channel gain denominator: ', channel_rate_denominator)
    #print('Achieved channel rate: ',channel_rate)

    achieved_transmission_delay = (offload_ratio*packet_size_bits)/(allocated_RB*channel_rate)
    achieved_transmission_energy_consumption = assigned_transmit_power_W*achieved_transmission_delay
    local_energy.append(achieved_local_energy_consumption)
    transmit_energy.append(achieved_transmission_energy_consumption)
    throughput.append(allocated_RB*channel_rate)
    total_energy.append((achieved_local_energy_consumption+achieved_transmission_energy_consumption))
    #print('achieved transmit delay: ', achieved_transmission_delay)
    print('achieved transmission energy consumption: ', achieved_transmission_energy_consumption)
            #self.achieved_transmission_energy_consumption 

    #print('total energy: ', (achieved_local_energy_consumption+achieved_transmission_energy_consumption))

# plotting
print('Max throughput: ', min(throughput), 'At power: ', transmit_powers[0])
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(transmit_powers, throughput, color ="red")
#plt.plot(allocated_RBs,local_energy,color = "blue")
#plt.plot(allocated_RBs,transmit_energy,color = "green")
#plt.plot(offload_ratios,throughput,color = "black")
plt.legend(["total throughput", "local computation energy", "Offloading energy"])
plt.xlabel("Transmit Power (dBm)")
plt.ylabel("Throughput (bits/s)")
plt.title("Throughput vs Transmit Power, Allocated RBs = 15, offloading ratio = 0.5")
plt.show()



    
