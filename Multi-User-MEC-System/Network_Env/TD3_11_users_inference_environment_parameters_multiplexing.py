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


# plt.figure(figsize=(15, 8))
# plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# # Subplot 1: Reward
# plt.subplot(3, 4, 1)
# plt.plot(task_arrival_rates, reward_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, reward_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Effect on Reward')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.legend(loc="lower left")

# # Subplot 2: Energy
# plt.subplot(3, 4, 2)
# plt.plot(task_arrival_rates, energy_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, energy_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Effect on Energy')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Energy (Joules)')
# plt.grid(True)
# plt.legend(loc="lower right")

# # Subplot 3: Throughput
# plt.subplot(3, 4, 3)
# plt.plot(task_arrival_rates, throughput_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, throughput_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Effect on Throughput')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Throughput')
# plt.grid(True)
# plt.legend(loc="lower right")

# # Subplot 4: Fairness Index
# plt.subplot(3, 4, 4)
# plt.plot(task_arrival_rates, fairness_index_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, fairness_index_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Effect on Fairness Index')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Fairness Index')
# plt.grid(True)
# plt.legend(loc="lower left")

# # Subplot 5: Delay
# plt.subplot(3, 4, 5)
# plt.plot(task_arrival_rates, delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, delay_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Effect on Delay')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Delay (ms)')
# plt.grid(True)
# plt.legend(loc="upper left")

# # Subplot 5: Delay
# plt.subplot(3, 4, 6)
# plt.plot(task_arrival_rates, offloading_ratios_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offloading_ratios_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Effect on Offloading Ratio')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Offloading Ratio')
# plt.grid(True)
# plt.legend(loc="lower left")

# plt.subplot(3, 4, 7)
# plt.plot(task_arrival_rates, local_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_delay_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Sum Local Delay')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Sum Local Delay')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(3, 4, 8)
# plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_delay_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Sum Offload Delay')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Sum Offload Delay')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(3, 4, 9)
# #plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, outage_probability_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Outage Probability')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Outage Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(3, 4, 10)
# #plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('Number of Dropped Transmissions')
# plt.xlabel('Task arrival rate')
# #plt.ylabel('Sum Offload Delay')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(3, 4, 11)
# #plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, urllc_throughput_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('URLLC Sum Data Rate')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Data Rate (bits/s)')
# plt.grid(True)
# plt.legend(loc="upper left")

#plt.tight_layout()
#plt.show()














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


import numpy as np
import matplotlib.pyplot as plt

# Load data
inf_number_of_arriving_urllc_packets = np.load('inf_number_of_arriving_urllc_packets.npy')
inf_number_of_dropped_urllc_packets_due_to_resource_allocation = np.load('inf_number_of_dropped_urllc_packets_due_to_resource_allocation.npy')
inf_number_of_dropped_urllc_packets_due_to_channel_rate = np.load('inf_number_of_dropped_urllc_packets_due_to_channel_rate.npy')
inf_outage_probability = np.load('inf_outage_probability.npy')
inf_failed_urllc_transmissions = np.load('inf_failed_urllc_transmissions.npy')
inf_total_urllc_data_rate = np.load('inf_total_urllc_data_rate.npy')
inf_urllc_successful_transmissions = np.load('inf_urllc_successful_transmissions.npy')

inf_individual_number_of_arriving_urllc_packets = np.load('inf_individual_number_of_arriving_urllc_packets.npy')
inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation = np.load('inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation.npy')
inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate = np.load('inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate.npy')
inf_individual_urllc_data_rate = np.load('inf_individual_urllc_data_rate.npy')
inf_individual_successful_transmissions = np.load('inf_individual_successful_transmissions.npy')

# inf_L_values = np.load('inf_L_values.npy')
# inf_cdf_values = np.load('inf_cdf_values.npy')

# Define timesteps
len_inf_individual_urllc_data_rate = len(inf_individual_urllc_data_rate)
timesteps = np.arange(1, len_inf_individual_urllc_data_rate + 1)

#max_index = 910#np.argmax(inf_total_urllc_data_rate)

# Specify the timestep at which to place markers
marker_timestep = 625#max_index  # Change this to any valid timestep
marker_index = np.where(timesteps == marker_timestep)[0][0]  # Get index in array

# Create subplots
plt.figure(figsize=(15, 8))

def plot_with_marker(position, y_data, title, ylabel=None):
    """Helper function to plot with a marker at the specified timestep and annotate it."""
    plt.subplot(3, 4, position)
    plt.plot(timesteps, y_data, label=title)
    
    # Get y-value at the marker point
    y_marker_value = y_data[marker_index]
    
    # Plot marker
    plt.plot(marker_timestep, y_marker_value, 'ro', markersize=8)  # Red circle marker
    
    # Annotate value next to the marker
    plt.text(marker_timestep, y_marker_value, f"{y_marker_value:.2f}", fontsize=12, fontweight='bold', verticalalignment='bottom', horizontalalignment='right', color='black')
    
    plt.title(title)
    plt.xlabel('Timestep')
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True)
    #plt.legend()

    # Print value at marker point
    print(f"{title} at timestep {marker_timestep}: {y_marker_value}")

# Plot each graph with markers and annotations
plot_with_marker(1, inf_outage_probability, 'Outage Probability vs Time')
plot_with_marker(2, inf_total_urllc_data_rate, 'Total URLLC Data Rate', ylabel='Data Rate (bits/s)')
plot_with_marker(3, inf_number_of_arriving_urllc_packets, 'Total Arriving Packets')
plot_with_marker(4, inf_failed_urllc_transmissions, 'Total Failed Transmissions')
plot_with_marker(5, inf_urllc_successful_transmissions, 'Total Successful Transmissions')
plot_with_marker(6, inf_number_of_dropped_urllc_packets_due_to_resource_allocation, 'Packets Dropped (Resource Allocation)')
plot_with_marker(7, inf_number_of_dropped_urllc_packets_due_to_channel_rate, 'Packets Dropped (Channel Rate)')
plot_with_marker(8, inf_individual_urllc_data_rate[:, 0], 'User 1: Channel Rate', ylabel='Channel Rate (bits/s)')
plot_with_marker(9, inf_individual_number_of_arriving_urllc_packets[:, 0], 'User 1: Arriving Packets')
plot_with_marker(10, inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation[:, 0], 'User 1: Dropped Packets (Resource Allocation)')
plot_with_marker(11, inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate[:, 0], 'User 1: Dropped Packets (Channel Rate)')
plot_with_marker(12, inf_individual_successful_transmissions[:, 0], 'User 1: Successful Transmission')
# Adjust layout and show plot
plt.tight_layout()
plt.show()

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


# plt.tight_layout()
# plt.show()



#-------------------------------------------------------------------------------------------------
task_arrival_rates = [0.1,0.2,0.3,0.4,0.5]

# throughput_policy_3_multiplexing = [23519968.206958,23388521.182457,23224525.972719,22983403.060768,22930902.268947]
# fairness_index_policy_3_multiplexing = [0.526857,0.529248,0.519982,0.527532,0.521553]
# outage_probability_policy_3_multiplexing = [0.002293,0.018301,0.065473,0.125250,0.253280]
# failed_urllc_transmissions_policy_3_multiplexing = [0.996040,2.000000,3.144554,4.029703,5.175248]
# urllc_throughput_policy_3_multiplexing = [2882056.549439,2879323.436229,2812820.270929,2903565.987765,2839120.937516]
# urllc_arriving_packets_policy_3_multiplexing = [1.629703,3.144554,4.800000,6.386139,8.041584]
# urllc_dropped_packets_resource_allocation_policy_3_multiplexing = [0.583168,1.205941,1.951485,2.380198,3.063366]
# urllc_dropped_packets_channel_rate_policy_3_multiplexing = [0.412871,0.794059,1.193069,1.649505,2.111881]
# urllc_successful_transmissions_policy_3_multiplexing = [0.615842,1.116832,1.606931,2.291089,2.791089]
# offloading_ratios_policy_3_multiplexing = [0.1937689188308986,0.19665145899048386,0.19201698283652677,0.19355456873137825,0.19152883704277743]

# throughput_policy_2_multiplexing = [24773791.471006,24549116.300087,24313272.036482,24421899.283341,24430706.705216]
# fairness_index_policy_2_multiplexing = [0.288577,0.288080,0.294306,0.287238,0.284221]
# outage_probability_policy_2_multiplexing = [0.008947,0.066277,0.177419,0.356919,0.536628]
# failed_urllc_transmissions_policy_2_multiplexing = [1.212871,2.358416,3.468317,4.708911,6.000000]
# urllc_throughput_policy_2_multiplexing = [2027907.910071,2049252.344470,2042827.509050,2041191.642018,2049137.276187]
# urllc_arriving_packets_policy_2_multiplexing = [1.638614,3.184158,4.700000,6.340594,8.033663]
# urllc_dropped_packets_resource_allocation_policy_2_multiplexing = [0.896040,1.769307,2.512871,3.471287,4.398020]
# urllc_dropped_packets_channel_rate_policy_2_multiplexing = [0.316832,0.589109,0.955446,1.237624,1.601980]
# urllc_successful_transmissions_policy_2_multiplexing = [0.402970,0.801980,1.187129,1.568317,1.953465]
# offloading_ratios_policy_2_multiplexing = [0.20004509703913625,0.19466166423448755,0.19373112978716292,0.1916621501716979,0.1952534814620856]

# throughput_policy_3_multiplexing = [26055989.413315,26141644.798566,25666780.073476,25616517.092900,25747475.643823]
# fairness_index_policy_3_multiplexing = [0.344939,0.339367,0.342452,0.339908,0.341447]
# outage_probability_policy_3_multiplexing = [0.001557,0.015538,0.044205,0.093633,0.193520]
# failed_urllc_transmissions_policy_3_multiplexing = [0.595758,0.637042,0.618786,0.625483,0.631657]
# urllc_throughput_policy_3_multiplexing = [3009658.962059,2959191.467821,2998482.614563,3023319.050738,2964893.941599]
# urllc_arriving_packets_policy_3_multiplexing = [1.587129,3.186139,4.680198,6.402970,7.975248]
# urllc_dropped_packets_resource_allocation_policy_3_multiplexing = [0.259513,0.286513,0.277131,0.278491,0.280695]
# urllc_dropped_packets_channel_rate_policy_3_multiplexing = [0.336245,0.350528,0.341654,0.346992,0.350962]
# urllc_dropped_packets_channel_rate_normalized_policy_3_multiplexing = [0.029126,0.030684,0.029799,0.030127,0.030685]
# urllc_successful_transmissions_policy_3_multiplexing = [0.393013,0.355500,0.371906,0.364466,0.359156]
# urllc_code_blocks_allocation_policy_3_multiplexing = [11.544554,11.423762,11.465347,11.517822,11.437624]
# offloading_ratios_policy_3_multiplexing = [0.19734782334083506,0.19152996122992938,0.19374182417445543,0.19257744881699448,0.1943328811035819]


# throughput_policy_2_multiplexing = [28976485.641096,29050629.857899,28535625.807299,28682629.958113,28560980.840404]
# fairness_index_policy_2_multiplexing = [0.462364,0.456957,0.464761,0.450561,0.465193]
# outage_probability_policy_2_multiplexing = [0.005134,0.046631,0.104546,0.249503,0.400373]
# failed_urllc_transmissions_policy_2_multiplexing = [0.717807,0.712054,0.692836,0.713707,0.713547]
# urllc_throughput_policy_2_multiplexing = [2329865.069970,2308734.550171,2415939.144192,2299263.781503,2344023.106419]
# urllc_arriving_packets_policy_2_multiplexing = [1.606931,3.104950,4.754455,6.349505,8.039604]
# urllc_dropped_packets_resource_allocation_policy_2_multiplexing = [0.475046,0.463329,0.445231,0.470763,0.458867]
# urllc_dropped_packets_channel_rate_policy_2_multiplexing = [0.242760,0.248724,0.247605,0.242944,0.254680]
# urllc_dropped_packets_channel_rate_normalized_policy_2_multiplexing = [0.028238,0.028868,0.028188,0.028813,0.029556]
# urllc_successful_transmissions_policy_2_multiplexing = [0.268022,0.277742,0.296335,0.276782,0.276970]
# urllc_code_blocks_allocation_policy_2_multiplexing = [8.597030,8.615842,8.784158,8.431683,8.616832]
# offloading_ratios_policy_2_multiplexing = [0.1962780915187864,0.19806346672530062,0.19648169967867882,0.1972022774722136,0.19630124660866402]

# throughput_policy_3_multiplexing = [26725752.891838,26790690.355098,26590930.079820,26518234.582759,26377871.419373]
# fairness_index_policy_3_multiplexing = [0.356295,0.351338,0.350647,0.352478,0.347640]
# outage_probability_policy_3_multiplexing = [0.041410,0.191288,0.370343,0.542379,0.693617]
# failed_urllc_transmissions_policy_3_multiplexing = [0.807094,0.789307,0.795863,0.798897,0.798490]
# urllc_throughput_policy_3_multiplexing = [1647890.446745,1594706.348482,1590595.634520,1615746.852710,1633131.945769]
# urllc_arriving_packets_policy_3_multiplexing = [1.591089,3.129703,4.738614,6.464356,7.866337]
# urllc_dropped_packets_resource_allocation_policy_3_multiplexing = [0.634723,0.605505,0.619933,0.614030,0.615607]
# urllc_dropped_packets_channel_rate_policy_3_multiplexing = [0.172371,0.183803,0.175930,0.184868,0.182882]
# urllc_dropped_packets_channel_rate_normalized_policy_3_multiplexing = [0.028139,0.030568,0.029570,0.030554,0.030147]
# urllc_successful_transmissions_policy_3_multiplexing = [0.181083,0.200253,0.192645,0.192219,0.192070]
# urllc_code_blocks_allocation_policy_3_multiplexing = [6.125743,6.012871,5.949505,6.050495,6.066337]
# offloading_ratios_policy_3_multiplexing = [0.19719305711514662,0.19697381942146988,0.1947635685919775,0.19613747617287966,0.19207713245669789]
# # ---------------------------------------
# # Average Individual Number of allocated RBs:  [0. 1. 0. 0. 3. 6. 1. 0. 1. 0. 0.]
# # ---------------------------------------
# # ---------------------------------------
# # Average Individual Number of Punctruring users:  [0 0 0 0 0 2 0 0 1 0 0]
# # ---------------------------------------
# # ---------------------------------------
# # Average Individual Number of Clustered Urllc users:  [0 0 6 2 0 2 0 3 1 1 1]


# throughput_policy_2_multiplexing = [28920487.464893,28782067.941722,28535289.100306,28486371.577977,28233539.833356]
# fairness_index_policy_2_multiplexing = [0.499652,0.503242,0.497064,0.497738,0.495541]
# outage_probability_policy_2_multiplexing = [0.007372,0.039645,0.099043,0.214785,0.365292]
# failed_urllc_transmissions_policy_2_multiplexing = [0.708435,0.710518,0.689994,0.708091,0.695776]
# urllc_throughput_policy_2_multiplexing = [2408479.466263,2445415.743503,2525250.323111,2476958.271838,2479741.117376]
# urllc_arriving_packets_policy_2_multiplexing = [1.619802,3.191089,4.739604,6.301980,7.993069]
# urllc_dropped_packets_resource_allocation_policy_2_multiplexing = [0.406479,0.392181,0.400877,0.413668,0.406045]
# urllc_dropped_packets_channel_rate_policy_2_multiplexing = [0.301956,0.318337,0.289116,0.294423,0.289731]
# urllc_dropped_packets_channel_rate_normalized_policy_2_multiplexing = [0.032455,0.033830,0.030232,0.031037,0.030610]
# urllc_successful_transmissions_policy_2_multiplexing = [0.283007,0.279243,0.299979,0.281854,0.295058]
# urllc_code_blocks_allocation_policy_2_multiplexing = [9.303960,9.409901,9.563366,9.486139,9.465347]
# offloading_ratios_policy_2_multiplexing = [0.19276356143727294,0.19510906104485243,0.19713557618672126,0.19977564373620196,0.19312052109603678]
# ---------------------------------------
# Average Individual Number of allocated RBs:  [1. 0. 2. 1. 1. 1. 1. 1. 1. 3. 0.]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Punctruring users:  [0 0 6 2 0 2 0 3 1 1 0]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Clustered Urllc users:  [0 0 6 2 0 2 0 3 1 1 1]



throughput_policy_3_multiplexing = [21941375.040442,21847321.270962,21674236.497227,21627541.901626,21370788.807311]
fairness_index_policy_3_multiplexing = [0.308369,0.303736,0.308661,0.312745,0.311955]
outage_probability_policy_3_multiplexing = [0.024452,0.099756,0.238773,0.390995,0.533596]
failed_urllc_transmissions_policy_3_multiplexing = [0.756246,0.757113,0.764422,0.760184,0.746760]
urllc_throughput_policy_3_multiplexing = [1908990.708027,1985442.121225,1955344.995418,1973403.863648,2037813.114767]
urllc_arriving_packets_policy_3_multiplexing = [1.624752,3.305941,4.925743,6.440594,8.022772]
urllc_dropped_packets_resource_allocation_policy_3_multiplexing = [0.560024,0.579515,0.593166,0.578017,0.571887]
urllc_dropped_packets_channel_rate_policy_3_multiplexing = [0.196222,0.177598,0.171256,0.182168,0.174874]
urllc_dropped_packets_channel_rate_normalized_policy_3_multiplexing = [0.029897,0.026523,0.025705,0.027181,0.025410]
urllc_successful_transmissions_policy_3_multiplexing = [0.232176,0.233603,0.226332,0.231360,0.244724]
urllc_code_blocks_allocation_policy_3_multiplexing = [6.563366,6.696040,6.662376,6.701980,6.882178]
offloading_ratios_policy_3_multiplexing = [0.1977619312388212,0.1959598971906783,0.19543719774182386,0.195986240837136,0.20049876094314464]
# ---------------------------------------
# Average Individual Number of allocated RBs:  [2. 0. 0. 0. 4. 4. 1. 0. 0. 0. 1.]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Punctruring users:  [0 0 0 0 0 3 1 0 0 0 2]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Clustered Urllc users:  [0 2 1 2 0 3 1 0 5 0 2]
# ---------------------------------------

throughput_policy_2_multiplexing = [19713888.547248,19644050.276370,19535506.358380,19328482.741731,19176758.832985]
fairness_index_policy_2_multiplexing = [0.501938,0.501944,0.500354,0.498758,0.501106]
outage_probability_policy_2_multiplexing = [0.003372,0.018057,0.052885,0.101842,0.194097]
failed_urllc_transmissions_policy_2_multiplexing = [0.613179,0.628376,0.614149,0.619650,0.626276]
urllc_throughput_policy_2_multiplexing = [3057720.583359,3004250.968804,3022033.252944,3070825.635528,3031453.289693]
urllc_arriving_packets_policy_2_multiplexing = [1.622772,3.189109,4.870297,6.338614,8.048515]
urllc_dropped_packets_resource_allocation_policy_2_multiplexing = [0.380720,0.401118,0.387071,0.388316,0.395006]
urllc_dropped_packets_channel_rate_policy_2_multiplexing = [0.232459,0.227259,0.227079,0.231334,0.231271]
urllc_dropped_packets_channel_rate_normalized_policy_2_multiplexing = [0.023953,0.023692,0.023642,0.023747,0.023974]
urllc_successful_transmissions_policy_2_multiplexing = [0.373398,0.361689,0.374060,0.371134,0.363267]
urllc_code_blocks_allocation_policy_2_multiplexing = [9.704950,9.592079,9.604950,9.741584,9.646535]
offloading_ratios_policy_2_multiplexing = [0.19439493772822347,0.19397932585356695,0.1961126629822247,0.19666448277437942,0.19427947397164191]

# ---------------------------------------
# Average Individual Number of allocated RBs:  [1. 2. 0. 0. 2. 0. 0. 2. 0. 2. 3.]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Punctruring users:  [0 2 0 0 0 0 0 0 0 0 2]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Clustered Urllc users:  [0 2 1 2 0 3 1 0 5 0 2]

plt.figure(figsize=(15, 8))
plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.subplot(3, 4, 1)
plt.plot(task_arrival_rates, throughput_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, throughput_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('eMBB Data Rate')
plt.xlabel('URLLC Task Arrival Probability')
plt.ylabel('Data Rate (bits/s)')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 5: Delay
plt.subplot(3, 4, 2)
plt.plot(task_arrival_rates, offloading_ratios_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, offloading_ratios_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('eMBB Offloading Ratios')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Delay (ms)')
plt.grid(True)
plt.legend(loc="upper left")

# Subplot 4: Fairness Index
plt.subplot(3, 4, 3)
plt.plot(task_arrival_rates, fairness_index_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, fairness_index_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('Effect on Fairness Index')
plt.xlabel('URLLC Task Arrival Probability')
plt.ylabel('Fairness Index')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 2: Energy
plt.subplot(3, 4, 4)
plt.plot(task_arrival_rates, urllc_throughput_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_throughput_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC Data Rate')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Energy (Joules)')
plt.grid(True)
plt.legend(loc="lower right")

# Subplot 3: Throughput
plt.subplot(3, 4, 5)
plt.plot(task_arrival_rates, outage_probability_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, outage_probability_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC Outage Probability')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Throughput')
plt.grid(True)
plt.legend(loc="lower right")


plt.subplot(3, 4, 6)
plt.plot(task_arrival_rates, urllc_arriving_packets_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_arriving_packets_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC Arriving Packets')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Local Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 7)
plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC Failed Transmissions')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 8)
plt.plot(task_arrival_rates, urllc_successful_transmissions_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_successful_transmissions_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC Successful Transmissions')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 9)
plt.plot(task_arrival_rates, urllc_dropped_packets_resource_allocation_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_dropped_packets_resource_allocation_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('Failed Transmissions (Resource Allocation)')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Outage Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 10)
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('Failed Transmissions (Channel Rate)')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 11)
plt.plot(task_arrival_rates, urllc_code_blocks_allocation_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_code_blocks_allocation_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC CBs allocation')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 12)
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_normalized_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^1$")
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_normalized_policy_2_multiplexing, marker='o', color='blue', label=r"$\pi_3^2$")
plt.title('URLLC Failed Transmissions (Channel Rate) Normalized')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")


plt.tight_layout()
plt.show()
# plt.subplot(3, 4, 11)
# #plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, urllc_throughput_policy_3_multiplexing, marker='o', color='blue', label=r"$\pi_3$ multiplexing")
# plt.title('URLLC Sum Data Rate')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Data Rate (bits/s)')
# plt.grid(True)
# plt.legend(loc="upper left")

# ---------------------------------------
# Average Individual Number of allocated RBs:  [2. 0. 0. 0. 4. 4. 1. 0. 0. 0. 1.]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Punctruring users:  [0 0 0 0 0 3 1 0 0 0 2]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Clustered Urllc users:  [0 2 1 2 0 3 1 0 5 0 2]
number_of_allocated_RBs_11_embb_users_3_multiplexing = [2, 0, 0, 0, 4, 4, 1, 0, 0, 0, 1]
number_of_clustered_urllc_users_11_embb_users_3_multiplexing = [0, 2, 1, 2, 0, 3, 1, 0, 5, 0, 2]



# Average Individual Number of allocated RBs:  [1. 2. 0. 0. 2. 0. 0. 2. 0. 2. 3.]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Punctruring users:  [1 0 0 0 0 3 1 0 0 1 0]
# ---------------------------------------
# ---------------------------------------
# Average Individual Number of Clustered Urllc users:  [1 1 0 0 1 7 1 3 1 1 0]

number_of_allocated_RBs_11_embb_users_2_multiplexing = [1, 2, 0, 0, 2, 0, 0, 2, 0, 2, 3]
number_of_clustered_urllc_users_11_embb_users_2_multiplexing = [0, 2, 1, 2, 0, 3, 1, 0, 5, 0, 2]


embb_users_11_users = ['1', '2', '3','4', '5', '6','7','8', '9', '10','11']

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
# Allocated RBs Plot
axes[0].bar(embb_users_11_users, number_of_allocated_RBs_11_embb_users_3_multiplexing, color='red')
axes[0].set_ylabel('Number of Allocated RBs')
axes[0].set_title('Number of Allocated RBs per eMBB User ($\pi_3^1$)')
axes[0].set_xlabel('eMBB User Index')
#axes[2].set_title('Allocated RBs per eMBB User')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Clustered URLLC Users Plot
axes[1].bar(embb_users_11_users, number_of_clustered_urllc_users_11_embb_users_3_multiplexing, color='red')
axes[1].set_ylabel('Number of Clustered URLLC Users')
axes[1].set_xlabel('eMBB User Index')
axes[1].set_title('Number of Clustered URLLC Users per eMBB User ($\pi_3^1$)')
#axes[1,1].set_xlabel('eMBB Users')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Allocated RBs Plot
axes[2].bar(embb_users_11_users, number_of_allocated_RBs_11_embb_users_2_multiplexing, color='blue')
axes[2].set_ylabel('Number of Allocated RBs')
axes[2].set_title('Number of Allocated RBs per eMBB User ($\pi_3^2$)')
axes[2].set_xlabel('eMBB User Index')
#axes[2].set_title('Allocated RBs per eMBB User')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

# Clustered URLLC Users Plot
axes[3].bar(embb_users_11_users, number_of_clustered_urllc_users_11_embb_users_2_multiplexing, color='blue')
axes[3].set_ylabel('Number of Clustered URLLC Users')
axes[3].set_xlabel('eMBB User Index')
axes[3].set_title('Number of Clustered URLLC Users per eMBB User ($\pi_3^2$)')
#axes[1,1].set_xlabel('eMBB Users')
axes[3].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



inf_L_values_0_2_p = np.load('inf_L_values_0_2_p.npy')
inf_cdf_values_0_2_p = np.load('inf_cdf_values_0_2_p.npy')

inf_L_values_0_5_p = np.load('inf_L_values_0_5_p.npy')
inf_cdf_values_0_5_p = np.load('inf_cdf_values_0_5_p.npy')

inf_L_values_0_8_p = np.load('inf_L_values_0_8_p.npy')
inf_cdf_values_0_8_p = np.load('inf_cdf_values_0_8_p.npy')

U = 16
p_0_5 = 0.5
p_0_2 = 0.2
p_0_8 = 0.8
# Plot CDF
plt.scatter(inf_L_values_0_2_p, inf_cdf_values_0_2_p,label=f'Binomial CDF (U={U}, p={p_0_2})', linewidth=2, color='blue')
plt.scatter(inf_L_values_0_5_p, inf_cdf_values_0_5_p,label=f'Binomial CDF (U={U}, p={p_0_5})', linewidth=2,color='red')
plt.scatter(inf_L_values_0_8_p, inf_cdf_values_0_8_p,label=f'Binomial CDF (U={U}, p={p_0_8})', linewidth=2, color='green')
#plt.xlabel('Total number of packet arrivals')
plt.xlabel(r'$\frac{\sum_{u \in U^{(m)}} R_u^{(m)} [t]}{l_u^{(m)}}$')

#plt.set_xlabel(r'$\frac{\sum_{u \in U^{(m)}} R_u^{(m)} [t]}{l_u^{(m)}}$')
plt.ylabel('Cumulative Probability')
plt.title(r'CDF of $\frac{\sum_{u \in U^{(m)}} R_u^{(m)} [t]}{l_u^{(m)}}$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()



