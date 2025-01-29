import numpy as np
import matplotlib.pyplot as plt

# Effecet of changing task arrival rate on reward, energy, throughput and fairness
import matplotlib.pyplot as plt

# # Data
# task_arrival_rates = [1, 5, 10, 15, 20]
# reward = [15670331.541776, -24946865.138353, -123207124.656152, -148644527.052450, -183945248.857225]
# energy = [0.000455, 0.000959, 0.001151, 0.001195, 0.001219]
# throughput = [26525919.029431, 31852185.114663, 25221962.110995, 30054260.612891, 28981972.082567]
# fairness_index = [0.499351, 0.523646, 0.525010, 0.507314, 0.533321]
# delay = [8.053156, 84.833167, 262.333365, 321.555117, 389.283486]
# offloading_ratios = [0.5223319215621657, 0.7082332962124606, 0.7535654580171486, 0.7844495787155158, 0.7878115681725816]

# # Plotting
# plt.figure(figsize=(10, 7))

# plt.suptitle('Effect of Task arrival rate on perfomance metrics',fontsize=16, fontweight='bold')

# # Subplot 1: Reward
# plt.subplot(3, 2, 1)
# plt.plot(task_arrival_rates, reward, marker='o', label='Reward', color='blue')
# plt.title('Effect on Reward')
# plt.xlabel('Task Arrival Rate')
# plt.ylabel('Reward')
# plt.grid(True)

# # Subplot 2: Energy
# plt.subplot(3, 2, 2)
# plt.plot(task_arrival_rates, energy, marker='o', label='Energy', color='orange')
# plt.title('Effect on Energy')
# plt.xlabel('Task Arrival Rate')
# plt.ylabel('Energy (Joules)')
# plt.grid(True)

# # Subplot 3: Throughput
# plt.subplot(3, 2, 3)
# plt.plot(task_arrival_rates, throughput, marker='o', label='Throughput', color='green')
# plt.title('Effect on Throughput')
# plt.xlabel('Task Arrival Rate')
# plt.ylabel('Throughput')
# plt.grid(True)

# # Subplot 4: Fairness Index
# plt.subplot(3, 2, 4)
# plt.plot(task_arrival_rates, fairness_index, marker='o', label='Fairness', color='red')
# plt.title('Effect on Fairness Index')
# plt.xlabel('Task Arrival Rate')
# plt.ylabel('Fairness Index')
# plt.grid(True)

# # Subplot 5: Delay
# plt.subplot(3, 2, 5)
# plt.plot(task_arrival_rates, delay, marker='o', label='Delay', color='purple')
# plt.title('Effect on Delay')
# plt.xlabel('Task Arrival Rate')
# plt.ylabel('Delay (ms)')
# plt.grid(True)

# # Subplot 6: Offloading Ratios
# plt.subplot(3, 2, 6)
# plt.plot(task_arrival_rates, offloading_ratios, marker='o', label='Offloading Ratios', color='brown')
# plt.title('Effect on Offloading Ratios')
# plt.xlabel('Task Arrival Rate')
# plt.ylabel('Offloading Ratios')
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# # Explanation:
# # Reward: A significant decline with increasing task arrival rates suggests inefficiencies or penalties under higher loads.
# # Energy: Gradual increase with task arrival rates, indicating higher energy consumption for processing or offloading.
# # Throughput: Variability suggests possible bottlenecks or inefficiencies at higher task rates.
# # Fairness Index: Generally improves, though fluctuates, indicating more equitable resource allocation at higher rates.
# # Delay: Steadily rises, highlighting the increased latency as the system becomes more burdened.
# # Offloading Ratios: Gradual increase, suggesting more tasks are offloaded to handle higher rates.


# # -------------------------------------------------------------------------------------------------------------------------------------------------------
# # Effect of changing gNB transmit power on Reward, energy, throughput, fairness and delay
# #  

# gnb_transmit_powers = [0,2,4,6,8,10] #x-axis
# reward = [-29375363.362543,-28732903.769288,-36351723.658050,-40174642.485284,-20920507.030925,-24163458.228770]#y-axis
# energy = [0.000962,0.000962,0.000955,0.000975,0.000958,0.000966]#y-axis
# throughput = [28503736.208997,29554877.212108,26490547.152658,25408570.123269,31689103.902675,29483598.697072]#y-axis
# fairness_index = [0.527945,0.524097,0.531061,0.528730,0.536614,0.523588]#y-axis
# delay = [86.885946,87.718288,97.036288,101.908512,76.471796,78.326663]#y-axis
# offloading_ratios = [0.7137059244797588,0.701895677842869,0.6964232672062806,0.6888176684764843,0.7122747627521058,0.7181269214686813]#y-axis
# energy_harvested = [0.0,0.0001640709497294605,0.00017624151601258224,0.00021167312870633067,0.000637361858202713,0.0008465833745685607]#y-axis
# battery_energy_levels = [26639.995633741353,26639.99899376778,26639.99971747164,26639.999840378237,26639.99991288652,26639.99991183396]#y-axis


# # Plotting
# plt.figure(figsize=(10, 7))
# plt.suptitle('Effect of gNB Transmission Power on perfomance metrics',fontsize=16, fontweight='bold')
# # Subplot 1: Reward
# plt.subplot(4, 2, 1)
# plt.plot(gnb_transmit_powers, reward, marker='o', color='blue', label='Reward')
# plt.title('Effect on Reward')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Reward')
# plt.grid(True)

# # Subplot 2: Energy
# plt.subplot(4, 2, 2)
# plt.plot(gnb_transmit_powers, energy, marker='o', color='orange', label='Energy')
# plt.title('Effect on Energy')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Energy (Joules)')
# plt.grid(True)

# # Subplot 3: Throughput
# plt.subplot(4, 2, 3)
# plt.plot(gnb_transmit_powers, throughput, marker='o', color='green', label='Throughput')
# plt.title('Effect on Throughput')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Throughput')
# plt.grid(True)

# # Subplot 4: Fairness Index
# plt.subplot(4, 2, 4)
# plt.plot(gnb_transmit_powers, fairness_index, marker='o', color='red', label='Fairness Index')
# plt.title('Effect on Fairness Index')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Fairness Index')
# plt.grid(True)

# # Subplot 5: Delay
# plt.subplot(4, 2, 5)
# plt.plot(gnb_transmit_powers, delay, marker='o', color='purple', label='Delay')
# plt.title('Effect on Delay')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Delay (ms)')
# plt.grid(True)

# # Subplot 6: Offloading Ratios
# plt.subplot(4, 2, 6)
# plt.plot(gnb_transmit_powers, offloading_ratios, marker='o', color='brown', label='Offloading Ratios')
# plt.title('Effect on Offloading Ratios')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Offloading Ratios')
# plt.grid(True)

# # Subplot 7: Energy Harvested
# plt.subplot(4, 2, 7)
# plt.plot(gnb_transmit_powers, energy_harvested, marker='o', color='cyan', label='Energy Harvested')
# plt.title('Effect on Energy Harvested')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Energy Harvested (Joules)')
# plt.grid(True)

# # Subplot 8: Battery Energy Levels
# plt.subplot(4, 2, 8)
# plt.plot(gnb_transmit_powers, battery_energy_levels, marker='o', color='magenta', label='Battery Energy Levels')
# plt.title('Effect on Battery Energy Levels')
# plt.xlabel('gNB Transmit Power (W)')
# plt.ylabel('Battery Energy Levels')
# plt.grid(True)

# plt.tight_layout()
# plt.show()



# # -------------------------------------------------------------------------------------------------------------------------------------------------------
# # Effect of changing number of users on Reward, energy, throughput, fairness and delay
# #  

# number_of_users = [3,7,11] #x-axis
# reward = [27061671.594047,24738542.987665,-62529259.317639]#y-axis
# energy = [0.000535,0.000701,0.000953]#y-axis
# throughput = [36792406.942830,40040386.882232,28761606.819416]#y-axis
# fairness_index = [0.668313,0.617915,0.503658]#y-axis
# delay = [3.410925,9.577313,154.003384]#y-axis

# # Plotting
# plt.figure(figsize=(15, 8))
# plt.suptitle('Effect of varying number of users on perfomance metrics',fontsize=16, fontweight='bold')
# # Subplot 1: Reward
# plt.subplot(3, 2, 1)
# plt.plot(number_of_users, reward, marker='o', color='blue', label='Reward')
# plt.title('Effect on Reward')
# plt.xlabel('Number of Users')
# plt.ylabel('Reward')
# plt.grid(True)

# # Subplot 2: Energy
# plt.subplot(3, 2, 2)
# plt.plot(number_of_users, energy, marker='o', color='orange', label='Energy')
# plt.title('Effect on Energy')
# plt.xlabel('Number of Users')
# plt.ylabel('Energy (Joules)')
# plt.grid(True)

# # Subplot 3: Throughput
# plt.subplot(3, 2, 3)
# plt.plot(number_of_users, throughput, marker='o', color='green', label='Throughput')
# plt.title('Effect on Throughput')
# plt.xlabel('Number of Users')
# plt.ylabel('Throughput')
# plt.grid(True)

# # Subplot 4: Fairness Index
# plt.subplot(3, 2, 4)
# plt.plot(number_of_users, fairness_index, marker='o', color='red', label='Fairness Index')
# plt.title('Effect on Fairness Index')
# plt.xlabel('Number of Users')
# plt.ylabel('Fairness Index')
# plt.grid(True)

# # Subplot 5: Delay
# plt.subplot(3, 2, 5)
# plt.plot(number_of_users, delay, marker='o', color='purple', label='Delay')
# plt.title('Effect on Delay')
# plt.xlabel('Number of Users')
# plt.ylabel('Delay (ms)')
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Effect of changing task arrival rate on Reward, energy, throughput, fairness, delay and other constraints
#  

# task_arrival_rates = [1, 5, 10, 15, 20]
# reward = [25150306.873748,19237812.040248,15124281.526113,3525548.033548,-19615508.326227]
# energy = [0.000672,0.000739,0.000840,0.000915,0.001012]
# throughput = [38626678.418975,35055288.356561,34867875.103429,34891645.654897,34903767.316802]
# fairness_index = [0.505567,0.515175,0.502585,0.509090,0.515719]
# delay = [6.790477,9.450226,14.276721,35.268336,78.668921]
# offloading_ratios = [0.8794278710760423,0.8822272267815386,0.8858810623893607,0.8815467953122659,0.8802795830339059]
# battery_energy_constraint = [0.0,0.0,0.0,0.0,0.0]
# local_traffic_intensity_constraint = [0.11386138613861387,0.12376237623762376,0.24752475247524752,0.44554455445544555,0.6683168316831684]
# offload_traffic_intensity_constraint = [3.6732673267326734,3.6683168316831685,4.445544554455446,5.272277227722772,6.178217821782178]
# rmin_constraint = [3.6831683168316833,3.5594059405940595,3.6633663366336635,3.485148514851485,3.5495049504950495]
# local_queue_delay_violation_probability_constraint = [0.01973497500732877,0.10437381557055962,0.20505082499712088,0.2851353105686306,0.3355265183573104]
# offload_queue_delay_violation_probability_constraint = [0.061580055612746036,0.2983827655602641,0.5864354473353665,0.7704808891937528,0.8596693651218752]

# task_arrival_rates = [1, 3, 5, 7, 9]
# reward_policy_1 = [13365236.530060,8562768.153210,-16888951.316286,-48499640.889443,-76240523.466994]
# energy_policy_1 = [0.000722,0.000778,0.000939,0.001024,0.001136]
# throughput_policy_1 = [28017677.513373,27876533.715575,28111920.079349,28167109.882637,28091342.818096]
# fairness_index_policy_1 = [0.530311,0.547120,0.545347,0.523858,0.525915]
# delay_policy_1 = [7.656474,15.281000,61.837919,122.603223,174.593483]
# local_delay_policy_1 = []
# offload_delay_policy_1 = []
# local_queue_length_tasks_policy_1 = []
# offload_queue_length_tasks_policy_1 = []
# local_queue_length_bits_policy_1 = []
# offload_queue_length_bits_policy_1 = []
# offloading_ratios_policy_1 = [0.8738168057401418,0.8845646295016615,0.8761979151655788,0.8779971139909777,0.877607942320634]
# battery_energy_constraint_policy_1 = [0.0,0.0,0.0,0.0,0.0]
# local_traffic_intensity_constraint_policy_1 = [0.0072007200720072,0.052205220522052204,0.13906390639063906,0.1984698469846985,0.25967596759675965]
# offload_traffic_intensity_constraint_policy_1 = [0.40594059405940597,0.45319531953195324,0.5414041404140414,0.6282628262826283,0.7227722772277227]
# local_queue_delay_violation_probability_constraint_policy_1 = [0.13546354635463545,0.3388838883888389,0.4126912691269127,0.427992799279928,0.46039603960396036]
# offload_queue_delay_violation_probability_constraint_policy_1 = [0.5063006300630063,0.986048604860486,0.9954995499549956,0.9972997299729974,0.9995499549954996]
# rmin_constraint_policy_1 = [0.328982898289829,0.32043204320432045,0.3195319531953195,0.3411341134113411,0.33753375337533753]
# local_queue_delay_violation_probability_policy_1 = [0.07326449000891345,0.20000689324641513,0.3049411835399544,0.344642792230233,0.3896012230530385]
# offload_queue_delay_violation_probability_policy_1 = [0.23465933429001073,0.6498917142398961,0.8510417192793062,0.9099704272102933,0.9294927637741527]

task_arrival_rates = [1, 3, 5, 7, 9]
reward_policy_1 = []
energy_policy_1 = []
throughput_policy_1 = []
fairness_index_policy_1 = []
delay_policy_1 = []

local_delay_policy_1 = []
offload_delay_policy_1 = []
local_queue_length_tasks_policy_1 = []
offload_queue_length_tasks_policy_1 = []
local_queue_length_bits_policy_1 = []
offload_queue_length_bits_policy_1 = []

offloading_ratios_policy_1 = []
battery_energy_constraint_policy_1 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_1 = []
offload_traffic_intensity_constraint_policy_1 = []
local_queue_delay_violation_probability_constraint_policy_1 = []
offload_queue_delay_violation_probability_constraint_policy_1 = []
rmin_constraint_policy_1 = []
local_queue_delay_violation_probability_policy_1 = []
offload_queue_delay_violation_probability_policy_1 = []


reward_policy_2 = []
energy_policy_2 = []
throughput_policy_2 = []
fairness_index_policy_2 = []
delay_policy_2 = []

local_delay_policy_2 = []
offload_delay_policy_2 = []
local_queue_length_tasks_policy_2 = []
offload_queue_length_tasks_policy_2 = []
local_queue_length_bits_policy_2 = []
offload_queue_length_bits_policy_2 = []

offloading_ratios_policy_2 = []
battery_energy_constraint_policy_2 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_2 = []
offload_traffic_intensity_constraint_policy_2 = []
local_queue_delay_violation_probability_constraint_policy_2 = []
offload_queue_delay_violation_probability_constraint_policy_2 = []
rmin_constraint_policy_2 = []
local_queue_delay_violation_probability_policy_2 = []
offload_queue_delay_violation_probability_policy_2 = []


reward_policy_3 = []
energy_policy_3 = []
throughput_policy_3 = []
fairness_index_policy_3 = []
delay_policy_3 = []

local_delay_policy_3 = []
offload_delay_policy_3 = []
local_queue_length_tasks_policy_3 = []
offload_queue_length_tasks_policy_3 = []
local_queue_length_bits_policy_3 = []
offload_queue_length_bits_policy_3 = []

offloading_ratios_policy_3 = []
battery_energy_constraint_policy_3 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_3 = []
offload_traffic_intensity_constraint_policy_3 = []
local_queue_delay_violation_probability_constraint_policy_3 = []
offload_queue_delay_violation_probability_constraint_policy_3 = []
rmin_constraint_policy_3 = []
local_queue_delay_violation_probability_policy_3 = []
offload_queue_delay_violation_probability_policy_3 = []


reward_policy_4 = []
energy_policy_4 = []
throughput_policy_4 = []
fairness_index_policy_4 = []
delay_policy_4 = []

local_delay_policy_4 = []
offload_delay_policy_4 = []
local_queue_length_tasks_policy_4 = []
offload_queue_length_tasks_policy_4 = []
local_queue_length_bits_policy_4 = []
offload_queue_length_bits_policy_4 = []

offloading_ratios_policy_4 = []
battery_energy_constraint_policy_4 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_4 = []
offload_traffic_intensity_constraint_policy_4 = []
local_queue_delay_violation_probability_constraint_policy_4 = []
offload_queue_delay_violation_probability_constraint_policy_4 = []
rmin_constraint_policy_4 = []
local_queue_delay_violation_probability_policy_4 = []
offload_queue_delay_violation_probability_policy_4 = []
# Plotting
# Plotting
plt.figure(figsize=(15, 8))
plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.subplot(3, 3, 1)
plt.plot(task_arrival_rates, reward_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, reward_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, reward_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, reward_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Effect on Reward')
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
plt.title('Effect on Energy')
plt.xlabel('Task arrival rate')
plt.ylabel('Energy (Joules)')
plt.grid(True)
plt.legend(loc="lower right")

# Subplot 3: Throughput
plt.subplot(3, 3, 3)
plt.plot(task_arrival_rates, throughput_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, throughput_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, throughput_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, throughput_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Effect on Throughput')
plt.xlabel('Task arrival rate')
plt.ylabel('Throughput')
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

# Subplot 5: Delay
plt.subplot(3, 3, 5)
plt.plot(task_arrival_rates, delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Effect on Delay')
plt.xlabel('Task arrival rate')
plt.ylabel('Delay (ms)')
plt.grid(True)
plt.legend(loc="upper left")

# Subplot 5: Delay
plt.subplot(3, 3, 6)
plt.plot(task_arrival_rates, offloading_ratios_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offloading_ratios_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offloading_ratios_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offloading_ratios_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Effect on Offloading Ratio')
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
plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')

plt.subplot(4, 3, 1)
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_length_tasks_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Local Queue Length Tasks')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue length')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 2)
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offload Queue Length Tasks')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue Length')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 3)
plt.plot(task_arrival_rates, local_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Local Queue Length bits')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue Length')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 4)
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offload Queue Length bits')
plt.xlabel('Task arrival rate')
plt.ylabel('Queue Length')
plt.grid(True)
plt.legend(loc="upper left")


plt.subplot(4, 3, 5)
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Violation Probability (Local Traffic Load Constraint)')
plt.xlabel('Task arrival rate')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 6)
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Violation Probability (Offload Traffic Load Constraint)')
plt.xlabel('Task arrival rate')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 7)
plt.plot(task_arrival_rates, rmin_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, rmin_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, rmin_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, rmin_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Violation Probability (Rmin Constraint)')
plt.xlabel('Task arrival rate')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")


plt.subplot(4, 3, 8)
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Local Queue Violation Probability')
plt.xlabel('Task arrival rate')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="upper left")

plt.subplot(4, 3, 9)
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offload Queue Violation Probability')
plt.xlabel('Task arrival rate')
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

