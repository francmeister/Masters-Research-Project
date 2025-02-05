import numpy as np
import matplotlib.pyplot as plt

# Effecet of changing task arrival rate on reward, energy, throughput and fairness
import matplotlib.pyplot as plt

inf_individual_urllc_data_rate = np.load('inf_individual_urllc_data_rate.npy')
inf_number_of_arriving_urllc_packets = np.load('inf_number_of_arriving_urllc_packets.npy')
inf_number_of_dropped_urllc_packets_due_to_resource_allocation = np.load('inf_number_of_dropped_urllc_packets_due_to_resource_allocation.npy')
inf_number_of_dropped_urllc_packets_due_to_channel_rate = np.load('inf_number_of_dropped_urllc_packets_due_to_channel_rate.npy')
inf_individual_number_of_arriving_urllc_packets = np.load('inf_individual_number_of_arriving_urllc_packets.npy')
inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation = np.load('inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation.npy')
inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate = np.load('inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate.npy')

len_inf_individual_urllc_data_rate = len(inf_individual_urllc_data_rate)
timesteps = np.arange(1,len_inf_individual_urllc_data_rate+1)


task_arrival_rates = [0.5,1,1.5,2]


reward_policy_3 = [120406880.730438,115631403.214210,115703772.692519,111264115.049406]
energy_policy_3 = [0.000260,0.000465,0.000612,0.000746]
throughput_policy_3 = [23942860.807337,23939533.548922,23620714.685404,23919279.379266]
fairness_index_policy_3 = [0.518933,0.517681,0.528367,0.513796]
delay_policy_3 = [9.176288,11.589089,18.300077,49.498144]

local_delay_policy_3 = [9.278937,11.560926,18.190901,49.888093]
offload_delay_policy_3 = [5.162383,5.116417,5.575642,5.800213]
local_queue_length_tasks_policy_3 = [21.54950495049505,27.094059405940595,40.386138613861384,110.94554455445545]
offload_queue_length_tasks_policy_3 = [15.450495049504951,14.782178217821782,15.46039603960396,16.76732673267327]
local_queue_length_bits_policy_3 = [2597.762376237624,6868.529702970297,15294.30693069307,54384.67326732673]
offload_queue_length_bits_policy_3 = [964.6336633663366,1807.7376237623762,2730.0940594059407,4104.4009900990095]

offloading_ratios_policy_3 = [0.20008110095505013,0.19790003830509206,0.19531207564011008,0.1955417758249463]
battery_energy_constraint_policy_3 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_3 = [0.0018001800180018,0.09675967596759677,0.2488748874887489,0.445994599459946]
offload_traffic_intensity_constraint_policy_3 = [0.0144014401440144,0.0144014401440144,0.016201620162016202,0.01935193519351935]
local_queue_delay_violation_probability_constraint_policy_3 = [0.6264626462646264,0.8838883888388839,0.927992799279928,0.9392439243924393]
offload_queue_delay_violation_probability_constraint_policy_3 = [0.0279027902790279,0.09135913591359136,0.18586858685868587,0.2488748874887489]
rmin_constraint_policy_3 = [0.33663366336633666,0.3370837083708371,0.32313231323132313,0.33753375337533753]
local_queue_delay_violation_probability_policy_3 = [0.1944998956825988,0.4611067120829111,0.7334329521290991,0.8463508981453859]
offload_queue_delay_violation_probability_policy_3 = [0.039057613937156135,0.07102155212779643,0.10239690164775272,0.13039265985815324]

reward_policy_3_multiplexing = [123120550.006151,118986568.012512,116360400.985094,112217860.958892]
energy_policy_3_multiplexing = [0.000275,0.000462,0.000607,0.000735]
throughput_policy_3_multiplexing = [20750543.871301,20735170.540757,20610656.602787,20568442.958388]
fairness_index_policy_3_multiplexing = [0.506633,0.508103,0.523637,0.501670]
delay_policy_3_multiplexing = [9.316206,11.583686,19.743871,46.561435]

local_delay_policy_3_multiplexing = [9.515965,11.584854,19.516155,46.839663]
offload_delay_policy_3_multiplexing = [5.305524,5.269475,5.532247,5.854407]

outage_probability_policy_3_multiplexing = [0.154228,0.142707,0.111797,0.128982]
failed_urllc_transmissions_policy_3_multiplexing = [4.787129,4.757426,4.797030,4.698020]
urllc_throughput_policy_3_multiplexing = [3188712.833269,3278159.813355,3380425.078356,3259507.563553]


local_queue_length_tasks_policy_3_multiplexing = [22.04950495049505,26.247524752475247,45.51485148514851,106.62376237623762]
offload_queue_length_tasks_policy_3_multiplexing = [14.910891089108912,15.826732673267326,14.712871287128714,16.797029702970296]
local_queue_length_bits_policy_3_multiplexing = [2780.0940594059407,6765.965346534654,17184.579207920793,51503.16336633663]
offload_queue_length_bits_policy_3_multiplexing = [940.7128712871287,1793.7425742574258,2704.960396039604,3998.8811881188117]

offloading_ratios_policy_3_multiplexing = [0.1957858466519976,0.187495642725929,0.18520894819974823,0.1913477825675525]
battery_energy_constraint_policy_3_multiplexing = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_3_multiplexing = [0.0018001800180018,0.09315931593159316,0.2407740774077408,0.46309630963096304]
offload_traffic_intensity_constraint_policy_3_multiplexing = [0.011701170117011701,0.01755175517551755,0.01755175517551755,0.024302430243024305]
local_queue_delay_violation_probability_constraint_policy_3_multiplexing = [0.6327632763276327,0.8955895589558956,0.9324932493249325,0.9441944194419442]
offload_queue_delay_violation_probability_constraint_policy_3_multiplexing = [0.0279027902790279,0.12016201620162016,0.20477047704770476,0.27362736273627364]
rmin_constraint_policy_3_multiplexing = [0.337983798379838,0.34608460846084604,0.3271827182718272,0.3487848784878488]
local_queue_delay_violation_probability_policy_3_multiplexing = [0.1955400291357195,0.46715615051392917,0.7431893491053893,0.8518512558763813]
offload_queue_delay_violation_probability_policy_3_multiplexing = [0.04042339074721322,0.08008560878116566,0.10951908492438976,0.14812792104461123]


plt.figure(figsize=(15, 8))
plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.subplot(3, 4, 1)
plt.plot(task_arrival_rates, reward_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, reward_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Effect on Reward')
plt.xlabel('Task arrival rate')
plt.ylabel('Reward')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 2: Energy
plt.subplot(3, 4, 2)
plt.plot(task_arrival_rates, energy_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, energy_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Effect on Energy')
plt.xlabel('Task arrival rate')
plt.ylabel('Energy (Joules)')
plt.grid(True)
plt.legend(loc="lower right")

# Subplot 3: Throughput
plt.subplot(3, 4, 3)
plt.plot(task_arrival_rates, throughput_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, throughput_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Effect on Throughput')
plt.xlabel('Task arrival rate')
plt.ylabel('Throughput')
plt.grid(True)
plt.legend(loc="lower right")

# Subplot 4: Fairness Index
plt.subplot(3, 4, 4)
plt.plot(task_arrival_rates, fairness_index_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, fairness_index_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Effect on Fairness Index')
plt.xlabel('Task arrival rate')
plt.ylabel('Fairness Index')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 5: Delay
plt.subplot(3, 4, 5)
plt.plot(task_arrival_rates, delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, delay_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Effect on Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Delay (ms)')
plt.grid(True)
plt.legend(loc="upper left")

# Subplot 5: Delay
plt.subplot(3, 4, 6)
plt.plot(task_arrival_rates, offloading_ratios_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offloading_ratios_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Effect on Offloading Ratio')
plt.xlabel('Task arrival rate')
plt.ylabel('Offloading Ratio')
plt.grid(True)
plt.legend(loc="lower left")

plt.subplot(3, 4, 7)
plt.plot(task_arrival_rates, local_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_delay_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Sum Local Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Sum Local Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 8)
plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_delay_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Sum Offload Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 9)
#plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, outage_probability_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Outage Probability')
plt.xlabel('Task arrival rate')
plt.ylabel('Outage Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 10)
#plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Number of Dropped Transmissions')
plt.xlabel('Task arrival rate')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 11)
#plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, urllc_throughput_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('URLLC Sum Data Rate')
plt.xlabel('Task arrival rate')
plt.ylabel('Data Rate (bits/s)')
plt.grid(True)
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()














# plt.figure(figsize=(15, 8))
# plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')

# plt.subplot(4, 3, 1)
# plt.plot(task_arrival_rates, local_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_length_tasks_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Local Queue Length Tasks')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Queue length')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 2)
# plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Offload Queue Length Tasks')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Queue Length')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 3)
# plt.plot(task_arrival_rates, local_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_length_bits_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Local Queue Length bits')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Queue Length')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 4)
# plt.plot(task_arrival_rates, offload_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_length_bits_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Offload Queue Length bits')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Queue Length')
# plt.grid(True)
# plt.legend(loc="upper left")


# plt.subplot(4, 3, 5)
# plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Local Traffic Load Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 6)
# plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Offload Traffic Load Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 7)
# plt.plot(task_arrival_rates, rmin_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, rmin_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Rmin Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")


# plt.subplot(4, 3, 8)
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Local Queue Violation Probability')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 9)
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Offload Queue Violation Probability')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 10)
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Local Queue Violation Probability Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 12)
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Offload Queue Violation Probability Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")



inf_number_of_arriving_urllc_packets = np.load('inf_number_of_arriving_urllc_packets.npy')
inf_number_of_dropped_urllc_packets_due_to_resource_allocation = np.load('inf_number_of_dropped_urllc_packets_due_to_resource_allocation.npy')
inf_number_of_dropped_urllc_packets_due_to_channel_rate = np.load('inf_number_of_dropped_urllc_packets_due_to_channel_rate.npy')
inf_outage_probability = np.load('inf_outage_probability.npy')
inf_failed_urllc_transmissions = np.load('inf_failed_urllc_transmissions.npy')
inf_total_urllc_data_rate = np.load('inf_total_urllc_data_rate.npy')

inf_individual_number_of_arriving_urllc_packets = np.load('inf_individual_number_of_arriving_urllc_packets.npy')
inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation = np.load('inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation.npy')
inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate = np.load('inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate.npy')
inf_individual_urllc_data_rate = np.load('inf_individual_urllc_data_rate.npy')



len_inf_individual_urllc_data_rate = len(inf_individual_urllc_data_rate)
timesteps = np.arange(1,len_inf_individual_urllc_data_rate+1)


plt.figure(figsize=(15, 8))
#plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')

plt.subplot(3, 4, 1)
plt.plot(timesteps, inf_outage_probability)
#plt.plot(task_arrival_rates, local_queue_length_tasks_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Outage Probability vs Time')
plt.xlabel('Timestep')
#plt.ylabel('Queue length')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 2)
plt.plot(timesteps, inf_failed_urllc_transmissions)
#plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Total failed transmissions')
plt.xlabel('Timestep')
#plt.ylabel('Queue Length')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 3)
plt.plot(timesteps, inf_total_urllc_data_rate)
#plt.plot(task_arrival_rates, local_queue_length_bits_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Total URLLC Data Rate')
plt.xlabel('Timestep')
plt.ylabel('Data Rate (bits/s)')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 4)
plt.plot(timesteps, inf_number_of_arriving_urllc_packets)
#plt.plot(task_arrival_rates, offload_queue_length_bits_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Total Arriving Packets')
plt.xlabel('Timestep')
#plt.ylabel('Queue Le')
plt.grid(True)
#plt.legend(loc="upper left")


plt.subplot(3, 4, 5)
plt.plot(timesteps, inf_number_of_dropped_urllc_packets_due_to_resource_allocation)
#plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Total Packets Dropped (Resource Allocation)')
plt.xlabel('Timestep')
#plt.ylabel('Violation Probability')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 6)
plt.plot(timesteps, inf_number_of_dropped_urllc_packets_due_to_channel_rate)
#plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('Total Packets Dropped (Insufficient Channel Rate)')
plt.xlabel('Timestep')
#plt.ylabel('Violation Probability')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 7)
plt.plot(timesteps, inf_individual_number_of_arriving_urllc_packets[:,0])
#plt.plot(task_arrival_rates, rmin_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('URLLC User 1 (Number of Arriving Packets)')
plt.xlabel('Timestep')
#plt.ylabel('Violation Probability')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 8)
plt.plot(timesteps, inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation[:,0])
#plt.plot(task_arrival_rates, rmin_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('URLLC User 1 (Dropped Packets due to Resource Allocation)')
plt.xlabel('Timestep')
#plt.ylabel('Violation Probability')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 9)
plt.plot(timesteps, inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate[:,0])
#plt.plot(task_arrival_rates, rmin_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('URLLC User 1 (Droppes Packets due to Channel Rate)')
plt.xlabel('Timestep')
plt.ylabel('Violation Probability')
plt.grid(True)
#plt.legend(loc="upper left")

plt.subplot(3, 4, 10)
plt.plot(timesteps, inf_individual_urllc_data_rate[:,0])
#plt.plot(task_arrival_rates, rmin_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
plt.title('URLLC User 1 (Channel Rate)')
plt.xlabel('Timestep')
plt.ylabel('Channel Rate (bits/s)')
plt.grid(True)
#plt.legend(loc="upper left")


# plt.subplot(4, 3, 8)
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Local Queue Violation Probability')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 9)
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Offload Queue Violation Probability')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 10)
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Local Queue Violation Probability Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 12)
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Violation Probability (Offload Queue Violation Probability Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")


plt.tight_layout()
plt.show()





