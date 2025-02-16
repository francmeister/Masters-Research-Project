import numpy as np
import matplotlib.pyplot as plt




task_arrival_rates = [0.1,0.2,0.3,0.4,0.5]

throughput_policy_3_multiplexing = [28064717.477865,28009210.424167,27869057.905819,27761783.680899,27671858.705877]
fairness_index_policy_3_multiplexing = [0.341673,0.342213,0.341450,0.340576,0.340555]
outage_probability_policy_3_multiplexing = [0.014263,0.070974,0.193900,0.363469,0.561006]
failed_urllc_transmissions_policy_3_multiplexing = [0.780855,0.780587,0.781475,0.780013,0.781020]
urllc_throughput_policy_3_multiplexing = [1950284.792611,1957479.106247,1949585.903391,1955622.035790,1949141.978085]
urllc_arriving_packets_policy_3_multiplexing = [1.599921,3.195386,4.794059,6.409386,8.010673]
urllc_dropped_packets_resource_allocation_policy_3_multiplexing = [0.458612,0.457318,0.458480,0.458328,0.459063]
urllc_dropped_packets_channel_rate_policy_3_multiplexing = [0.322244,0.323269,0.322995,0.321685,0.321957]
urllc_dropped_packets_channel_rate_normalized_policy_3_multiplexing = [0.037665,0.037672,0.037689,0.037540,0.037575]
urllc_successful_transmissions_policy_3_multiplexing = [0.209515,0.209436,0.208406,0.210267,0.209092]
urllc_code_blocks_allocation_policy_3_multiplexing = [8.555624,8.581129,8.570079,8.569149,8.568337]
offloading_ratios_policy_3_multiplexing = [0.195289,0.196079,0.194961,0.195148,0.195915]
throughput_var_policy_3_multiplexing = [19914151764125.972656,19911298439142.878906,19799910786272.457031,19893663921384.789062,19784906583241.542969]
offload_ratio_var_policy_3_multiplexing = [0.000002,0.000003,0.000001,0.000001,0.000001]
urllc_throughput_var_policy_3_multiplexing = [488004742048.544250,483942774203.408691,487003217036.277283,484561348821.967590,485444135623.296021]
urllc_successful_transmissions_var_policy_3_multiplexing = [0.341616,0.672253,1.006118,1.367116,1.705128]
throughput_policy_5_multiplexing = [25264867.738203,25148768.289542,25001960.152954,24923468.811713,24774037.103254]
fairness_index_policy_5_multiplexing = [0.492068,0.490998,0.490036,0.488671,0.488932]
outage_probability_policy_5_multiplexing = [0.010727,0.051837,0.138236,0.268979,0.423319]
failed_urllc_transmissions_policy_5_multiplexing = [0.735228,0.734185,0.734113,0.734079,0.733449]
urllc_throughput_policy_5_multiplexing = [2290696.598066,2293006.519852,2291175.512040,2290084.359958,2294623.960120]
urllc_arriving_packets_policy_5_multiplexing = [1.603624,3.195624,4.802911,6.399050,7.996535]
urllc_dropped_packets_resource_allocation_policy_5_multiplexing = [0.396972,0.396539,0.394860,0.394923,0.395518]
urllc_dropped_packets_channel_rate_policy_5_multiplexing = [0.338256,0.337646,0.339254,0.339156,0.337931]
urllc_dropped_packets_channel_rate_normalized_policy_5_multiplexing = [0.035192,0.035159,0.035281,0.035314,0.035177]
urllc_successful_transmissions_policy_5_multiplexing = [0.254473,0.255417,0.256033,0.255870,0.256750]
urllc_code_blocks_allocation_policy_5_multiplexing = [9.611703,9.603366,9.615644,9.603980,9.606653]
offloading_ratios_policy_5_multiplexing = [0.195352,0.195789,0.194929,0.194744,0.195136]
throughput_var_policy_5_multiplexing = [18363892560026.105469,18362706775204.136719,18165171907213.589844,18094064634991.554688,18132133518700.472656]
offload_ratio_var_policy_5_multiplexing = [0.000002,0.000001,0.000002,0.000001,0.000001]
urllc_throughput_var_policy_5_multiplexing = [795023137067.091797,794635597456.914673,790539147828.447632,793908801590.388428,793020095511.069336]
urllc_successful_transmissions_var_policy_5_multiplexing = [0.415729,0.845927,1.311791,1.776488,2.263437]
throughput_policy_6_multiplexing = [24974953.754794,24831050.912414,24747456.540214,24632567.928944,24492053.436640]
fairness_index_policy_6_multiplexing = [0.447766,0.447599,0.446781,0.447379,0.446431]
outage_probability_policy_6_multiplexing = [0.014002,0.068195,0.171375,0.320287,0.483193]
failed_urllc_transmissions_policy_6_multiplexing = [0.747505,0.747236,0.747071,0.749536,0.748016]
urllc_throughput_policy_6_multiplexing = [2146201.142978,2143581.883287,2149411.316200,2136939.416798,2147645.668176]
urllc_arriving_packets_policy_6_multiplexing = [1.599485,3.211941,4.811743,6.417089,8.001287]
urllc_dropped_packets_resource_allocation_policy_6_multiplexing = [0.453500,0.452137,0.453680,0.454131,0.452274]
urllc_dropped_packets_channel_rate_policy_6_multiplexing = [0.294005,0.295099,0.293391,0.295406,0.295742]
urllc_dropped_packets_channel_rate_normalized_policy_6_multiplexing = [0.033878,0.034012,0.033799,0.034111,0.034050]
urllc_successful_transmissions_policy_6_multiplexing = [0.242368,0.242998,0.243229,0.240604,0.242045]
urllc_code_blocks_allocation_policy_6_multiplexing = [8.678317,8.676455,8.680416,8.660099,8.685545]
offloading_ratios_policy_6_multiplexing = [0.194540,0.195523,0.195045,0.195152,0.195632]
throughput_var_policy_6_multiplexing = [18956600908716.441406,18812721367672.179688,18790599351672.335938,18624956354041.972656,18676177281811.414062]
offload_ratio_var_policy_6_multiplexing = [0.000001,0.000002,0.000003,0.000001,0.000001]
urllc_throughput_var_policy_6_multiplexing = [739130638750.043579,741067729910.176270,751182445440.167480,739748127924.038208,738451686572.673462]
urllc_successful_transmissions_var_policy_6_multiplexing = [0.395242,0.804431,1.258682,1.700858,2.148980]

plt.figure(figsize=(15, 8))
plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.subplot(3, 4, 1)
plt.plot(task_arrival_rates, throughput_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, throughput_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, throughput_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('eMBB Data Rate')
plt.xlabel('URLLC Task Arrival Probability')
plt.ylabel('Data Rate (bits/s)')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 5: Delay
plt.subplot(3, 4, 2)
plt.plot(task_arrival_rates, offloading_ratios_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, offloading_ratios_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, offloading_ratios_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('eMBB Offloading Ratios')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Delay (ms)')
plt.grid(True)
plt.legend(loc="upper left")

# Subplot 4: Fairness Index
plt.subplot(3, 4, 3)
plt.plot(task_arrival_rates, fairness_index_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, fairness_index_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, fairness_index_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")

plt.title('Effect on Fairness Index')
plt.xlabel('URLLC Task Arrival Probability')
plt.ylabel('Fairness Index')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 2: Energy
plt.subplot(3, 4, 4)
plt.plot(task_arrival_rates, urllc_throughput_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_throughput_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_throughput_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Data Rate')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Energy (Joules)')
plt.grid(True)
plt.legend(loc="lower right")

# Subplot 3: Throughput
plt.subplot(3, 4, 5)
plt.plot(task_arrival_rates, outage_probability_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, outage_probability_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, outage_probability_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Outage Probability')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Throughput')
plt.grid(True)
plt.legend(loc="lower right")


plt.subplot(3, 4, 6)
plt.plot(task_arrival_rates, urllc_arriving_packets_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_arriving_packets_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_arriving_packets_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Arriving Packets')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Local Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 7)
plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, failed_urllc_transmissions_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Failed Transmissions')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 8)
plt.plot(task_arrival_rates, urllc_successful_transmissions_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_successful_transmissions_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_successful_transmissions_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Successful Transmissions')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 9)
plt.plot(task_arrival_rates, urllc_dropped_packets_resource_allocation_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_dropped_packets_resource_allocation_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_dropped_packets_resource_allocation_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('Failed Transmissions (Resource Allocation)')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Outage Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 10)
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('Failed Transmissions (Channel Rate)')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 11)
plt.plot(task_arrival_rates, urllc_code_blocks_allocation_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_code_blocks_allocation_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_code_blocks_allocation_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC CBs allocation')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 4, 12)
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_normalized_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_normalized_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_dropped_packets_channel_rate_normalized_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Failed Transmissions (Channel Rate) Normalized')
plt.xlabel('URLLC Task Arrival Probability')
#plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")


plt.tight_layout()
plt.show()


throughput_var_policy_3_multiplexing = [19914151764125.972656,19911298439142.878906,19799910786272.457031,19893663921384.789062,19784906583241.542969]
offload_ratio_var_policy_3_multiplexing = [0.000002,0.000003,0.000001,0.000001,0.000001]
urllc_throughput_var_policy_3_multiplexing = [488004742048.544250,483942774203.408691,487003217036.277283,484561348821.967590,485444135623.296021]
urllc_successful_transmissions_var_policy_3_multiplexing = [0.341616,0.672253,1.006118,1.367116,1.705128]

plt.figure(figsize=(15, 8))
plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.subplot(2, 2, 1)
plt.plot(task_arrival_rates, throughput_var_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, throughput_var_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, throughput_var_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('eMBB Users Throughput Variance')
plt.xlabel('URLLC Task Arrival Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(2, 2, 2)
plt.plot(task_arrival_rates, offload_ratio_var_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, offload_ratio_var_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, offload_ratio_var_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('Offload Ratio Variance')
plt.xlabel('URLLC Task Arrival Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(2, 2, 3)
plt.plot(task_arrival_rates, urllc_throughput_var_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_throughput_var_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_throughput_var_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('URLLC Users Throughtput Variance')
plt.xlabel('URLLC Task Arrival Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(2, 2, 4)
plt.plot(task_arrival_rates, urllc_successful_transmissions_var_policy_3_multiplexing, marker='o', color='red', label=r"$\pi_3^{mul}$")
plt.plot(task_arrival_rates, urllc_successful_transmissions_var_policy_5_multiplexing, marker='o', color='blue', label=r"$\pi_5^{mul}$")
plt.plot(task_arrival_rates, urllc_successful_transmissions_var_policy_6_multiplexing, marker='o', color='green', label=r"$\pi_6^{mul}$")
plt.title('Successful Transmissions Variance')
plt.xlabel('URLLC Task Arrival Probability')
plt.grid(True)
plt.legend(loc="upper left")



plt.tight_layout()
plt.show()