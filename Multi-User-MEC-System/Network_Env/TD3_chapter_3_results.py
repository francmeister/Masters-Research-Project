import numpy as np
import matplotlib.pyplot as plt

# Effecet of changing task arrival rate on reward, energy, throughput and fairness
import matplotlib.pyplot as plt


task_arrival_rates = [0.5,1,1.5,2]

reward_policy_1 = [127244015.949426,122840044.463361,115437786.894748,105651406.893061]
energy_policy_1 = [0.000630,0.000682,0.000724,0.000773]
throughput_policy_1 = [31694872.960144,31644647.753144,31665714.591565,31717297.372611]
fairness_index_policy_1 = [0.521646,0.520213,0.520834,0.520483]
delay_policy_1 = [8.315135,8.803960,9.776810,11.171782]
local_delay_policy_1 = [4.628701,5.029861,5.673808,6.540427]
offload_delay_policy_1 = [8.287076,8.589911,9.206907,10.100302]
local_queue_length_tasks_policy_1 = [10.528119,11.140693,12.134554,13.421188]
offload_queue_length_tasks_policy_1 = [29.736634,30.586733,32.594307,35.228168]
local_queue_length_bits_policy_1 = [512.719703,1376.153762,2317.396634,3551.412921]
offload_queue_length_bits_policy_1 = [2952.823317,7594.045198,12135.460743,17543.379901]
offloading_ratios_policy_1 = [0.805000,0.804362,0.804300,0.805066]
battery_energy_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_1 = [0.000000,0.010675,0.037385,0.072408]
offload_traffic_intensity_constraint_policy_1 = [0.323920,0.339662,0.364797,0.395369]
local_queue_delay_violation_probability_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_1 = [0.327457,0.327381,0.326733,0.327993]
local_queue_delay_violation_probability_policy_1 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_1 = [0.000000,0.000000,0.000000,0.000000]
reward_policy_2 = [127229115.556320,122877244.947212,115616113.772249,105407731.361385]
energy_policy_2 = [0.000630,0.000680,0.000726,0.000771]
throughput_policy_2 = [31696883.271884,31683118.704071,31675271.591408,31711873.863641]
fairness_index_policy_2 = [0.522011,0.520533,0.521270,0.521833]
delay_policy_2 = [8.325207,8.811219,9.762991,11.181384]
local_delay_policy_2 = [4.647270,5.006679,5.658448,6.514775]
offload_delay_policy_2 = [8.301185,8.599745,9.193303,10.104092]
local_queue_length_tasks_policy_2 = [10.534010,11.102030,12.101634,13.455050]
offload_queue_length_tasks_policy_2 = [29.712376,30.652030,32.515990,35.350248]
local_queue_length_bits_policy_2 = [506.696832,1359.095050,2304.466485,3556.841485]
offload_queue_length_bits_policy_2 = [2951.683614,7616.657376,12102.737574,17702.219752]
offloading_ratios_policy_2 = [0.802857,0.802627,0.801545,0.802035]
battery_energy_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_2 = [0.000014,0.010815,0.037795,0.074446]
offload_traffic_intensity_constraint_policy_2 = [0.323852,0.338591,0.365468,0.393227]
local_queue_delay_violation_probability_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_2 = [0.327417,0.327385,0.326962,0.326706]
local_queue_delay_violation_probability_policy_2 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_2 = [0.000000,0.000000,0.000000,0.000000]
reward_policy_3 = [126608862.376681,122186303.624699,117140299.052213,109544210.763598]
energy_policy_3 = [0.000240,0.000446,0.000614,0.000744]
throughput_policy_3 = [25241607.600374,25222851.412683,25238570.543941,25215686.248471]
fairness_index_policy_3 = [0.501608,0.502073,0.502799,0.503145]
delay_policy_3 = [9.148299,11.492365,19.923149,52.080493]
local_delay_policy_3 = [9.268389,11.474696,19.838251,52.357710]
offload_delay_policy_3 = [5.213091,5.328571,5.468058,5.716620]
local_queue_length_tasks_policy_3 = [21.388663,26.473218,44.676436,114.596535]
offload_queue_length_tasks_policy_3 = [14.843911,15.205000,15.475099,15.982871]
local_queue_length_bits_policy_3 = [2121.576436,6680.006535,16956.936535,57430.206089]
offload_queue_length_bits_policy_3 = [714.958564,1821.186089,2813.741683,3903.125743]
offloading_ratios_policy_3 = [0.195033,0.195016,0.195370,0.195751]
battery_energy_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_3 = [0.000144,0.097916,0.245720,0.447372]
offload_traffic_intensity_constraint_policy_3 = [0.323618,0.325792,0.327808,0.333258]
local_queue_delay_violation_probability_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_3 = [0.335189,0.334950,0.333222,0.333920]
local_queue_delay_violation_probability_policy_3 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_3 = [0.000000,0.000000,0.000000,0.000000]
reward_policy_4 = [127084820.956039,118022238.552520,102155051.897424,98200658.145945]
energy_policy_4 = [0.000177,0.000290,0.000383,0.000473]
throughput_policy_4 = [25907294.480509,25814481.705264,25836931.091089,25850075.652495]
fairness_index_policy_4 = [0.495130,0.497049,0.495671,0.494636]
delay_policy_4 = [8.513768,9.631659,11.963355,16.438550]
local_delay_policy_4 = [7.185328,8.096341,10.063401,14.093521]
offload_delay_policy_4 = [7.717353,8.148757,8.874031,9.971334]
local_queue_length_tasks_policy_4 = [16.427376,18.298762,22.306337,30.538564]
offload_queue_length_tasks_policy_4 = [26.730297,28.216683,30.340743,33.683317]
local_queue_length_bits_policy_4 = [1126.268911,3242.798614,6163.506832,11482.750743]
offload_queue_length_bits_policy_4 = [2114.550891,5560.618267,9037.267525,13504.897822]
offloading_ratios_policy_4 = [0.569423,0.569780,0.569143,0.569817]
battery_energy_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_4 = [0.000068,0.038623,0.108974,0.202066]
offload_traffic_intensity_constraint_policy_4 = [0.327412,0.340009,0.359514,0.380896]
local_queue_delay_violation_probability_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_4 = [0.336548,0.335149,0.335855,0.336832]
local_queue_delay_violation_probability_policy_4 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_4 = [0.000000,0.000000,0.000000,0.000000]
# Plotting
# Plotting
#plt.figure(figsize=(15, 8))
#plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.figure()
plt.subplot(3, 3, 1)
plt.plot(task_arrival_rates, reward_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, reward_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, reward_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, reward_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Total System Reward')
plt.xlabel('Task arrival rate')
plt.ylabel('Reward')
plt.grid(True)
plt.legend(loc="lower left")

# Subplot 2: Energy
plt.subplot(3, 3, 2)
plt.plot(task_arrival_rates, energy_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, energy_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, energy_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, energy_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Sum Energy Consumption')
plt.xlabel('Task arrival rate')
plt.ylabel('Energy (J)')
plt.grid(True)
plt.legend(loc="lower right")

#Subplot 3: Throughput
plt.subplot(3, 3, 3)
plt.plot(task_arrival_rates, throughput_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, throughput_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, throughput_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, throughput_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Sum Data Rate')
plt.xlabel('Task arrival rate')
plt.ylabel('Throughput (bits/s)')
plt.grid(True)
plt.legend(loc="lower right")

# Subplot 4: Fairness Index
plt.subplot(3, 3, 4)
plt.plot(task_arrival_rates, fairness_index_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, fairness_index_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, fairness_index_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, fairness_index_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Effect on Fairness Index')
plt.xlabel('Task arrival rate')
plt.ylabel('Fairness Index')
plt.grid(True)
plt.legend(loc="lower left")

#Subplot 5: Delay
plt.subplot(3, 3, 5)
plt.plot(task_arrival_rates, delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Sum Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Delay (ms)')
plt.grid(True)
plt.legend(loc="upper left")

# # Subplot 5: Delay
plt.subplot(3, 3, 6)
plt.plot(task_arrival_rates, offloading_ratios_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offloading_ratios_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offloading_ratios_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offloading_ratios_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offloading Ratios')
plt.xlabel('Task arrival rate')
plt.ylabel('Offloading Ratio')
plt.grid(True)
plt.legend(loc="lower left")

plt.subplot(3, 3, 7)
plt.plot(task_arrival_rates, local_delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Sum Local Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Sum Local Delay')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(3, 3, 8)
plt.plot(task_arrival_rates, offload_delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Sum Offload Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Sum Offload Delay')
plt.grid(True)
plt.legend(loc="upper left")


plt.tight_layout()
plt.show()














plt.figure(figsize=(15, 8))
#plt.figure()
#plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')

plt.subplot(4, 3, 1)
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Local Queue Length (Number of Tasks)')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue length (Tasks)')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 2)
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offload Queue Length (Number of Tasks)')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue Length (Tasks)')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 3)
plt.plot(task_arrival_rates, local_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Local Queue Length (Number of bits)')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue Length (bits)')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 4)
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offload Queue Length (Number of bits)')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue Length (bits)')
plt.grid(True)
plt.legend(loc="upper left")


plt.subplot(4, 3, 5)
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title(r'Local Traffic Load Constraint Violation Probability ($\Pr \left( \rho_{d,lc}^{(m)} > 1 \right) $)')
plt.xlabel(r'Task arrival rate ($\lambda$)')
plt.ylabel('Violation Probability ')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 6)
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title(r'Offload Traffic Load Constraint Violation Probability ($\Pr \left( \rho_{d,off}^{(m)} > 1 \right) $)')
plt.xlabel(r'Task arrival rate ($\lambda$)')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 7)
plt.plot(task_arrival_rates, rmin_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, rmin_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, rmin_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, rmin_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title(r'$R^{\min}$ Constraint Violation Probability ($\Pr\left(R_d^{(m)}[t] < R^{\min}\right)$)')
plt.xlabel(r'Task arrival rate ($\lambda$)')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")


plt.subplot(4, 3, 8)
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title(r'Local Queue Delay Violation Probability ($\Pr\left( L_{d,lc}^{(m)}[t] > L_{d,lc}^{\max} \right)$)')
plt.xlabel(r'Task arrival rate ($\lambda$)')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 9)
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title(r'Offload Queue Delay Violation Probability ($\Pr\left( L_{d,off}^{(m)}[t] > L_{d,off}^{\max} \right)$)')
plt.xlabel(r'Task arrival rate ($\lambda$)')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 10)
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Violation Probability (Local Queue Violation Probability Constraint)')
plt.xlabel('Task arrival rate')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 12)
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Violation Probability (Offload Queue Violation Probability Constraint)')
plt.xlabel('Task arrival rate')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()

