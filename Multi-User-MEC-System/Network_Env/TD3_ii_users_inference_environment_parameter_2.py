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

# task_arrival_rates = [0.5,1,1.5,2]
# reward_policy_1 = [22664453.512069,21950106.415544,22051064.740691,20596195.356199]
# energy_policy_1 = [0.000643,0.000679,0.000673,0.000753]
# throughput_policy_1 = [33155397.495768,32997624.088812,33084987.462967,32999720.661115]
# fairness_index_policy_1 = [0.540252,0.537193,0.541332,0.541619]
# delay_policy_1 = [8.450150,8.604777,9.374369,11.125236]

# local_delay_policy_1 = [4.529290,4.839203,5.257661,6.720981]
# offload_delay_policy_1 = [8.375621,8.376204,8.959722,10.167736]
# local_queue_length_tasks_policy_1 = [10.311881188118813,10.727722772277227,11.351485148514852,13.564356435643564]
# offload_queue_length_tasks_policy_1 = [29.519801980198018,30.138613861386137,31.633663366336634,36.2029702970297]
# local_queue_length_bits_policy_1 = [596.9752475247525,1346.59900990099,1963.8217821782177,3398.9059405940593]
# offload_queue_length_bits_policy_1 = [3775.6188118811883,7408.30198019802,11605.475247524753,18619.980198019803]

# offloading_ratios_policy_1 = [0.806480654897737,0.803030811403164,0.8135609386927954,0.8077671050168181]
# battery_energy_constraint_policy_1 = [0.0,0.0,0.0,0.0,0.0]
# local_traffic_intensity_constraint_policy_1 = [0.0,0.011701170117011701,0.0279027902790279,0.0747074707470747]
# offload_traffic_intensity_constraint_policy_1 = [0.3258325832583258,0.33933393339333934,0.34743474347434744,0.38343834383438347]
# local_queue_delay_violation_probability_constraint_policy_1 = [0.0441044104410441,0.2344734473447345,0.30648064806480646,0.36993699369936994]
# offload_queue_delay_violation_probability_constraint_policy_1 = [0.054455445544554455,0.306030603060306,0.6314131413141314,0.7988298829882988]
# rmin_constraint_policy_1 = [0.32853285328532855,0.3343834383438344,0.32043204320432045,0.3159315931593159]
# local_queue_delay_violation_probability_policy_1 = [0.04555198154144394,0.10973667281806061,0.164399379029232,0.2228819320304784]
# offload_queue_delay_violation_probability_policy_1 = [0.09266843282214739,0.19195587635150357,0.29640381484559053,0.3836486151814218]

task_arrival_rates = [0.5,1,1.5,2]
reward_policy_1 = [20562546.004329,19383165.741631,18094618.219263,17294200.253931]
energy_policy_1 = [0.000617,0.000670,0.000740,0.000816]
throughput_policy_1 = [30661999.527245,30325341.402560,30217885.127168,30668203.881581]
fairness_index_policy_1 = [0.542546,0.535551,0.527767,0.520980]
delay_policy_1 = [8.393160,8.867115,10.249109,11.291349]

local_delay_policy_1 = [4.596535,5.172383,5.883074,6.487152]
offload_delay_policy_1 = [8.337669,8.609002,9.544934,10.050414]
local_queue_length_tasks_policy_1 = [10.232673267326733,11.42079207920792,12.881188118811881,13.01980198019802]
offload_queue_length_tasks_policy_1 = [29.143564356435643,30.816831683168317,33.81683168316832,36.301980198019805]
local_queue_length_bits_policy_1 = [630.6683168316831,1465.2326732673268,2525.262376237624,3707.019801980198]
offload_queue_length_bits_policy_1 = [3650.668316831683,7428.188118811881,12589.524752475247,18495.029702970296]

offloading_ratios_policy_1 = [0.8054139242269694,0.7926765550177465,0.8041783706847577,0.8009450906083451]
battery_energy_constraint_policy_1 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_1 = [0.0,0.010351035103510351,0.045004500450045004,0.07695769576957695]
offload_traffic_intensity_constraint_policy_1 = [0.0144014401440144,0.018901890189018902,0.025202520252025205,0.04050405040504051]
local_queue_delay_violation_probability_constraint_policy_1 = [0.05040504050405041,0.24752475247524755,0.3159315931593159,0.36003600360036003]
offload_queue_delay_violation_probability_constraint_policy_1 = [0.041854185418541856,0.3550855085508551,0.6962196219621962,0.8226822682268226]
rmin_constraint_policy_1 = [0.31863186318631864,0.3177317731773177,0.32853285328532855,0.3361836183618362]
local_queue_delay_violation_probability_policy_1 = [0.04586322199656987,0.11563697539462132,0.17288715187981346,0.230247643073214]
offload_queue_delay_violation_probability_policy_1 = [0.08951533420041938,0.1931044172235559,0.2976995875629108,0.3885035851526241]


# reward_policy_2 = [23074803.722140,22006247.384657,21463685.844513,20343884.021973]
# energy_policy_2 = [0.000617,0.000702,0.000714,0.000770]
# throughput_policy_2 = [33175581.947135,33409956.016396,33136064.633964,32936215.121599]
# fairness_index_policy_2 = [0.540396,0.535565,0.562775,0.544490]
# delay_policy_2 = [8.399237,8.803550,9.620942,10.457518]

# local_delay_policy_2 = [4.544059,5.220666,5.610744,6.314392]
# offload_delay_policy_2 = [8.344454,8.579167,9.088449,9.566787]
# local_queue_length_tasks_policy_2 = [10.51980198019802,11.683168316831683,11.782178217821782,13.287128712871286]
# offload_queue_length_tasks_policy_2 = [29.5,30.65841584158416,30.405940594059405,32.81188118811881]
# local_queue_length_bits_policy_2 = [629.4257425742575,1466.7821782178219,2266.079207920792,3320.227722772277]
# offload_queue_length_bits_policy_2 = [3668.4950495049507,7816.940594059406,11928.282178217822,16674.73267326733]

# offloading_ratios_policy_2 = [0.807684364562162,0.8079771152627402,0.8017097353036745,0.7966212828107608]
# battery_energy_constraint_policy_2 = [0.0,0.0,0.0,0.0,0.0]
# local_traffic_intensity_constraint_policy_2 = [0.0,0.011701170117011701,0.036453645364536456,0.06975697569756976]
# offload_traffic_intensity_constraint_policy_2 = [0.3136813681368137,0.3424842484248425,0.33663366336633666,0.37623762376237624]
# local_queue_delay_violation_probability_constraint_policy_2 = [0.05085508550855086,0.22637263726372636,0.3096309630963096,0.37353735373537356]
# offload_queue_delay_violation_probability_constraint_policy_2 = [0.05670567056705671,0.30063006300630063,0.6273627362736274,0.797929792979298]
# rmin_constraint_policy_2 = [0.31638163816381637,0.328982898289829,0.31098109810981095,0.3271827182718272]
# local_queue_delay_violation_probability_policy_2 = [0.04529850034766551,0.10670327289325225,0.17509861417391315,0.2351700725005093]
# offload_queue_delay_violation_probability_policy_2 = [0.09061171267737694,0.19117451684330392,0.28157816003493696,0.3772687941714731]

reward_policy_2 = [129891720.399378,126865235.985283,122493365.744130,113938688.709555]
energy_policy_2 = [0.000633,0.000696,0.000743,0.000747]
throughput_policy_2 = [30775449.792568,30641825.472521,30700466.677254,30527034.611330]
fairness_index_policy_2 = [0.529240,0.528282,0.539407,0.532648]
delay_policy_2 = [8.368185,8.940183,9.635180,10.959571]

local_delay_policy_2 = [4.866337,5.260302,5.388508,6.348945]
offload_delay_policy_2 = [8.329544,8.599921,9.246426,10.159120]
local_queue_length_tasks_policy_2 = [11.01980198019802,11.316831683168317,11.46039603960396,13.331683168316832]
offload_queue_length_tasks_policy_2 = [29.282178217821784,30.742574257425744,31.84158415841584,35.56930693069307]
local_queue_length_bits_policy_2 = [631.509900990099,1526.3465346534654,2054.628712871287,3319.7227722772277]
offload_queue_length_bits_policy_2 = [3660.509900990099,7562.336633663366,11944.40099009901,17266.980198019803]

offloading_ratios_policy_2 = [0.7954018595318096,0.7972334537689527,0.8091916987519514,0.8041943308047071]
battery_energy_constraint_policy_2 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_2 = [0.0,0.01395139513951395,0.031053105310531057,0.06795679567956796]
offload_traffic_intensity_constraint_policy_2 = [0.0144014401440144,0.018901890189018902,0.019801980198019802,0.040054005400540056]
local_queue_delay_violation_probability_constraint_policy_2 = [0.05265526552655265,0.2335733573357336,0.3114311431143114,0.3622862286228623]
offload_queue_delay_violation_probability_constraint_policy_2 = [0.04275427542754275,0.34518451845184517,0.693069306930693,0.8343834383438344]
rmin_constraint_policy_2 = [0.3271827182718272,0.3280828082808281,0.3096309630963096,0.32673267326732675]
local_queue_delay_violation_probability_policy_2 = [0.048186407715592496,0.11303150719743395,0.1682338332801015,0.2261024119617718]
offload_queue_delay_violation_probability_policy_2 = [0.0897935029628107,0.193390372237476,0.2974191776404587,0.39388708161746494]


# reward_policy_3 = [21145727.338697,18843577.948225,15482000.597588,10343774.517052]
# energy_policy_3 = [0.000281,0.000448,0.000600,0.000753]
# throughput_policy_3 = [26291134.333552,26747251.321035,26413678.219919,26856785.793485]
# fairness_index_policy_3 = [0.531974,0.535117,0.548796,0.538865]
# delay_policy_3 = [9.329636,11.818974,19.242750,52.139480]

# local_delay_policy_3 = [9.455457,11.766942,19.237966,52.568531]
# offload_delay_policy_3 = [5.034315,5.173216,5.309965,5.369316]
# local_queue_length_tasks_policy_3 = [22.321782178217823,26.955445544554454,41.96534653465346,112.99009900990099]
# offload_queue_length_tasks_policy_3 = [14.673267326732674,14.361386138613861,14.990099009900991,14.683168316831683]
# local_queue_length_bits_policy_3 = [2816.1435643564355,6983.341584158416,16045.356435643564,57601.38613861386]
# offload_queue_length_bits_policy_3 = [842.7920792079208,1616.8960396039604,2619.079207920792,3406.0346534653463]

# offloading_ratios_policy_3 = [0.1804988627989738,0.19268740153953023,0.1927818889871272,0.19826374964007912]
# battery_energy_constraint_policy_3 = [0.0,0.0,0.0,0.0,0.0]
# local_traffic_intensity_constraint_policy_3 = [0.0031503150315031507,0.10126012601260127,0.24752475247524755,0.43699369936993704]
# offload_traffic_intensity_constraint_policy_3 = [0.32403240324032406,0.3159315931593159,0.31863186318631864,0.32268226822682267]
# local_queue_delay_violation_probability_constraint_policy_3 = [0.648964896489649,0.8874887488748875,0.9311431143114312,0.9383438343834384]
# offload_queue_delay_violation_probability_constraint_policy_3 = [0.031053105310531057,0.09315931593159316,0.15796579657965795,0.22007200720072007]
# rmin_constraint_policy_3 = [0.33123312331233123,0.32088208820882086,0.31638163816381637,0.3213321332133213]
# local_queue_delay_violation_probability_policy_3 = [0.19938768380457741,0.46416032484487746,0.7358172623774812,0.8454193660720616]
# offload_queue_delay_violation_probability_policy_3 = [0.036385834001984696,0.06873575299368642,0.09535889187548174,0.12794442103232864]

reward_policy_3 = []
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


reward_policy_4 = [238690227.937680,231742378.374977,218909146.614664,200331171.881067]
energy_policy_4 = [0.000421,0.000498,0.000575,0.000682]
throughput_policy_4 = [29223619.528394,28385084.282162,28578499.568684,28652910.668562]
fairness_index_policy_4 = [0.495018,0.494225,0.493584,0.497754]
delay_policy_4 = [8.422536,9.184460,11.223810,15.513649]

local_delay_policy_4 = [7.345875,8.017798,9.304933,12.845240]
offload_delay_policy_4 = [7.763086,7.969824,8.803438,10.370458]
local_queue_length_tasks_policy_4 = [16.925742574257427,17.792079207920793,20.485148514851485,27.752475247524753]
offload_queue_length_tasks_policy_4 = [31.613861386138613,32.08910891089109,37.227722772277225,42.84653465346535]
local_queue_length_bits_policy_4 = [1360.8465346534654,3032.668316831683,5488.519801980198,9892.29207920792]
offload_queue_length_bits_policy_4 = [3027.7178217821784,6106.782178217822,11309.60396039604,17947.4900990099]

offloading_ratios_policy_4 = [0.5870597739895137,0.5750782165914236,0.5904967468641175,0.5895112332894267]
battery_energy_constraint_policy_4 = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_4 = [0.0009000900090009,0.030603060306030605,0.09360936093609361,0.17956795679567958]
offload_traffic_intensity_constraint_policy_4 = [0.01755175517551755,0.0288028802880288,0.06525652565256525,0.0891089108910891]
local_queue_delay_violation_probability_constraint_policy_4 = [0.17866786678667868,0.5135013501350135,0.5877587758775878,0.63996399639964]
offload_queue_delay_violation_probability_constraint_policy_4 = [0.07290729072907291,0.2961296129612961,0.4981998199819982,0.6264626462646264]
rmin_constraint_policy_4 = [0.37713771377137717,0.3757875787578758,0.37533753375337536,0.36858685868586855]
local_queue_delay_violation_probability_policy_4 = [0.09857329647893295,0.23995144401110893,0.3663530232497785,0.46366523887453637]
offload_queue_delay_violation_probability_policy_4 = [0.0867679730441355,0.17670344276534125,0.27986775298498556,0.34912213374582995]
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

