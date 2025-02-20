import numpy as np
import matplotlib.pyplot as plt

# Effecet of changing task arrival rate on reward, energy, throughput and fairness
import matplotlib.pyplot as plt


task_arrival_rates = [0.5,1,1.5,2]

# reward_policy_1 = [127244015.949426,122840044.463361,115437786.894748,105651406.893061]
# energy_policy_1 = [0.000630,0.000682,0.000724,0.000773]
# throughput_policy_1 = [31694872.960144,31644647.753144,31665714.591565,31717297.372611]
# fairness_index_policy_1 = [0.521646,0.520213,0.520834,0.520483]
# delay_policy_1 = [8.315135,8.803960,9.776810,11.171782]
# local_delay_policy_1 = [4.628701,5.029861,5.673808,6.540427]
# offload_delay_policy_1 = [8.287076,8.589911,9.206907,10.100302]
# local_queue_length_tasks_policy_1 = [10.528119,11.140693,12.134554,13.421188]
# offload_queue_length_tasks_policy_1 = [29.736634,30.586733,32.594307,35.228168]
# local_queue_length_bits_policy_1 = [512.719703,1376.153762,2317.396634,3551.412921]
# offload_queue_length_bits_policy_1 = [2952.823317,7594.045198,12135.460743,17543.379901]
# offloading_ratios_policy_1 = [0.805000,0.804362,0.804300,0.805066]
# battery_energy_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
# local_traffic_intensity_constraint_policy_1 = [0.000000,0.010675,0.037385,0.072408]
# offload_traffic_intensity_constraint_policy_1 = [0.323920,0.339662,0.364797,0.395369]
# local_queue_delay_violation_probability_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
# rmin_constraint_policy_1 = [0.327457,0.327381,0.326733,0.327993]
# local_queue_delay_violation_probability_policy_1 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_policy_1 = [0.000000,0.000000,0.000000,0.000000]
# reward_policy_2 = [127229115.556320,122877244.947212,115616113.772249,105407731.361385]
# energy_policy_2 = [0.000630,0.000680,0.000726,0.000771]
# throughput_policy_2 = [31696883.271884,31683118.704071,31675271.591408,31711873.863641]
# fairness_index_policy_2 = [0.522011,0.520533,0.521270,0.521833]
# delay_policy_2 = [8.325207,8.811219,9.762991,11.181384]
# local_delay_policy_2 = [4.647270,5.006679,5.658448,6.514775]
# offload_delay_policy_2 = [8.301185,8.599745,9.193303,10.104092]
# local_queue_length_tasks_policy_2 = [10.534010,11.102030,12.101634,13.455050]
# offload_queue_length_tasks_policy_2 = [29.712376,30.652030,32.515990,35.350248]
# local_queue_length_bits_policy_2 = [506.696832,1359.095050,2304.466485,3556.841485]
# offload_queue_length_bits_policy_2 = [2951.683614,7616.657376,12102.737574,17702.219752]
# offloading_ratios_policy_2 = [0.802857,0.802627,0.801545,0.802035]
# battery_energy_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
# local_traffic_intensity_constraint_policy_2 = [0.000014,0.010815,0.037795,0.074446]
# offload_traffic_intensity_constraint_policy_2 = [0.323852,0.338591,0.365468,0.393227]
# local_queue_delay_violation_probability_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
# rmin_constraint_policy_2 = [0.327417,0.327385,0.326962,0.326706]
# local_queue_delay_violation_probability_policy_2 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_policy_2 = [0.000000,0.000000,0.000000,0.000000]
# reward_policy_3 = [126608862.376681,122186303.624699,117140299.052213,109544210.763598]
# energy_policy_3 = [0.000240,0.000446,0.000614,0.000744]
# throughput_policy_3 = [25241607.600374,25222851.412683,25238570.543941,25215686.248471]
# fairness_index_policy_3 = [0.501608,0.502073,0.502799,0.503145]
# delay_policy_3 = [9.148299,11.492365,19.923149,52.080493]
# local_delay_policy_3 = [9.268389,11.474696,19.838251,52.357710]
# offload_delay_policy_3 = [5.213091,5.328571,5.468058,5.716620]
# local_queue_length_tasks_policy_3 = [21.388663,26.473218,44.676436,114.596535]
# offload_queue_length_tasks_policy_3 = [14.843911,15.205000,15.475099,15.982871]
# local_queue_length_bits_policy_3 = [2121.576436,6680.006535,16956.936535,57430.206089]
# offload_queue_length_bits_policy_3 = [714.958564,1821.186089,2813.741683,3903.125743]
# offloading_ratios_policy_3 = [0.195033,0.195016,0.195370,0.195751]
# battery_energy_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
# local_traffic_intensity_constraint_policy_3 = [0.000144,0.097916,0.245720,0.447372]
# offload_traffic_intensity_constraint_policy_3 = [0.323618,0.325792,0.327808,0.333258]
# local_queue_delay_violation_probability_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
# rmin_constraint_policy_3 = [0.335189,0.334950,0.333222,0.333920]
# local_queue_delay_violation_probability_policy_3 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_policy_3 = [0.000000,0.000000,0.000000,0.000000]
# reward_policy_4 = [127084820.956039,118022238.552520,102155051.897424,98200658.145945]
# energy_policy_4 = [0.000177,0.000290,0.000383,0.000473]
# throughput_policy_4 = [25907294.480509,25814481.705264,25836931.091089,25850075.652495]
# fairness_index_policy_4 = [0.495130,0.497049,0.495671,0.494636]
# delay_policy_4 = [8.513768,9.631659,11.963355,16.438550]
# local_delay_policy_4 = [7.185328,8.096341,10.063401,14.093521]
# offload_delay_policy_4 = [7.717353,8.148757,8.874031,9.971334]
# local_queue_length_tasks_policy_4 = [16.427376,18.298762,22.306337,30.538564]
# offload_queue_length_tasks_policy_4 = [26.730297,28.216683,30.340743,33.683317]
# local_queue_length_bits_policy_4 = [1126.268911,3242.798614,6163.506832,11482.750743]
# offload_queue_length_bits_policy_4 = [2114.550891,5560.618267,9037.267525,13504.897822]
# offloading_ratios_policy_4 = [0.569423,0.569780,0.569143,0.569817]
# battery_energy_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
# local_traffic_intensity_constraint_policy_4 = [0.000068,0.038623,0.108974,0.202066]
# offload_traffic_intensity_constraint_policy_4 = [0.327412,0.340009,0.359514,0.380896]
# local_queue_delay_violation_probability_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
# rmin_constraint_policy_4 = [0.336548,0.335149,0.335855,0.336832]
# local_queue_delay_violation_probability_policy_4 = [0.000000,0.000000,0.000000,0.000000]
# offload_queue_delay_violation_probability_policy_4 = [0.000000,0.000000,0.000000,0.000000]

reward_policy_1 = [129452913.399964,125926827.853698,120158959.158585,111636700.885171]
energy_policy_1 = [0.000629,0.000680,0.000725,0.000768]
throughput_policy_1 = [33813033.700656,33799557.270661,33827521.665518,33837925.172460]
fairness_index_policy_1 = [0.534127,0.533984,0.534195,0.534737]
normalized_fairness_index_policy_1 = [x - 0.18657 for x in fairness_index_policy_1]#[((x - min(fairness_index_policy_1)) / (max(fairness_index_policy_1) - min(fairness_index_policy_1))) * (0.347805 - 0.335195) + 0.335195 for x in fairness_index_policy_1]

delay_policy_1 = [8.318719,8.709475,9.559579,10.800815]
local_delay_policy_1 = [4.652517,5.029033,5.642009,6.499203]
offload_delay_policy_1 = [8.291362,8.498291,8.974460,9.699447]
local_queue_length_tasks_policy_1 = [10.593663,11.169347,12.045406,13.377485]
offload_queue_length_tasks_policy_1 = [29.719545,30.426891,31.869485,34.005307]
local_queue_length_bits_policy_1 = [515.255921,1375.045287,2313.377406,3536.298950]
offload_queue_length_bits_policy_1 = [2951.969743,7549.457248,11908.001228,16944.752772]
offloading_ratios_policy_1 = [0.804850,0.804639,0.805224,0.804884]
battery_energy_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_1 = [0.000013,0.010709,0.037069,0.072495]
offload_traffic_intensity_constraint_policy_1 = [0.322706,0.333554,0.355273,0.381667]
local_queue_delay_violation_probability_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_1 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_1 = [0.326009,0.326364,0.326144,0.325973]
local_queue_delay_violation_probability_policy_1 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_1 = [0.000000,0.000000,0.000000,0.000000]
var_reward_policy_1 = [655194579398415.875000,691748105560486.125000,770885774797285.625000,5909657299669411.000000]
var_energy_policy_1 = [0.000000,0.000000,0.000000,0.000000]
var_throughput_policy_1 = [20477118756439.285156,20646854296768.562500,20651369096887.476562,20669508474534.796875]
var_delay_policy_1 = [4.026523,4.770193,6.646762,10.143863]
var_local_queue_length_bits_policy_1 = [125751.009079,852604.561810,2310089.772990,4984215.653064]
var_offload_queue_length_bits_policy_1 = [1009796.783679,6630384.050667,16379441.458296,32959984.687057]
reward_policy_2 = [129440110.562350,126021316.739004,120259009.747531,111983519.819069]
energy_policy_2 = [0.000630,0.000680,0.000725,0.000768]
throughput_policy_2 = [33805156.950208,33787974.331208,33806635.755593,33824986.360362]
fairness_index_policy_2 = [0.534747,0.534860,0.534681,0.534600]
normalized_fairness_index_policy_2 = [x - 0.18657 for x in fairness_index_policy_2]#[((x - min(fairness_index_policy_2)) / (max(fairness_index_policy_2) - min(fairness_index_policy_2))) * (0.347805 - 0.335195) + 0.335195 for x in fairness_index_policy_2]

delay_policy_2 = [8.307064,8.707913,9.557044,10.787515]
local_delay_policy_2 = [4.654906,5.028701,5.661639,6.480976]
offload_delay_policy_2 = [8.283175,8.495581,8.965492,9.695789]
local_queue_length_tasks_policy_2 = [10.594970,11.165743,12.078832,13.341347]
offload_queue_length_tasks_policy_2 = [29.700554,30.375109,31.850990,33.940713]
local_queue_length_bits_policy_2 = [510.642653,1370.740416,2314.877426,3505.964475]
offload_queue_length_bits_policy_2 = [2936.155564,7527.022178,11861.271960,16918.410178]
offloading_ratios_policy_2 = [0.802082,0.802373,0.802288,0.802307]
battery_energy_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_2 = [0.000005,0.011030,0.038034,0.073703]
offload_traffic_intensity_constraint_policy_2 = [0.322675,0.332974,0.354549,0.381600]
local_queue_delay_violation_probability_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_2 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_2 = [0.325984,0.326079,0.326211,0.325890]
local_queue_delay_violation_probability_policy_2 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_2 = [0.000000,0.000000,0.000000,0.000000]
var_reward_policy_2 = [665633314637304.625000,699255855881248.125000,772347988957793.750000,1562921885189158.000000]
var_energy_policy_2 = [0.000000,0.000000,0.000000,0.000000]
var_throughput_policy_2 = [20654154089254.113281,20579496914811.164062,20627155265650.214844,20517183792097.433594]
var_delay_policy_2 = [4.038306,4.764034,6.670371,10.152724]
var_local_queue_length_bits_policy_2 = [122776.031155,856603.761190,2304973.491906,4961900.331451]
var_offload_queue_length_bits_policy_2 = [1007968.621027,6534310.799865,16432842.004651,33375463.614843]
reward_policy_3 = [128823259.075680,124749469.139487,120106271.813071,113288822.008518]
energy_policy_3 = [0.000240,0.000448,0.000614,0.000740]
throughput_policy_4 = [27342404.774491,27353594.149844,27329312.713181,27333752.856559]
fairness_index_policy_3 = [0.522149,0.523484,0.523950,0.522200]
delay_policy_3 = [9.123974,11.491121,20.027862,51.134646]
local_delay_policy_3 = [9.250622,11.487091,19.992606,51.413119]
offload_delay_policy_3 = [5.174362,5.269927,5.383023,5.570725]
local_queue_length_tasks_policy_3 = [21.350376,26.469822,45.143010,112.727465]
offload_queue_length_tasks_policy_3 = [14.771941,15.115069,15.381723,15.681604]
local_queue_length_bits_policy_3 = [2120.580257,6674.675307,17165.420475,56626.553248]
offload_queue_length_bits_policy_3 = [714.040158,1810.404139,2768.163406,3773.876178]
offloading_ratios_policy_3 = [0.194843,0.195781,0.195574,0.194807]
normalized_fairness_index_policy_3 = [x - 0.18657 for x in fairness_index_policy_3]#[((x - min(fairness_index_policy_3)) / (max(fairness_index_policy_3) - min(fairness_index_policy_3))) * (0.347805 - 0.335195) + 0.335195 for x in fairness_index_policy_3]

battery_energy_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_3 = [0.000133,0.098691,0.247123,0.447078]
offload_traffic_intensity_constraint_policy_3 = [0.323026,0.323989,0.326346,0.330929]
local_queue_delay_violation_probability_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_3 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_3 = [0.330401,0.329636,0.329329,0.330092]
local_queue_delay_violation_probability_policy_3 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_3 = [0.000000,0.000000,0.000000,0.000000]
var_reward_policy_3 = [655870266470375.000000,725100973203920.000000,790074157738120.875000,691662487304836.125000]
var_energy_policy_3 = [0.000000,0.000000,0.000000,0.000000]
var_throughput_policy_3 = [16613050724997.849609,16554449597632.136719,16561396792935.160156,16663584769094.421875]
var_delay_policy_3 = [3.819339,7.311141,40.176245,560.543631]
var_local_queue_length_bits_policy_3 = [550960.462014,5379507.927307,38762571.202725,657934648.276234]
var_offload_queue_length_bits_policy_3 = [187262.091734,1212731.490474,2734604.830170,4962756.150629]
reward_policy_4 = [128682026.678289,123475598.708259,116392205.560572,107232732.425604]
energy_policy_4 = [0.000177,0.000289,0.000381,0.000470]
throughput_policy_3 = [26897508.372835,26882734.101525,26908288.655873,26911513.332610]
fairness_index_policy_4 = [0.527417,0.526928,0.527267,0.528004]
normalized_fairness_index_policy_4 = [x - 0.18657 for x in fairness_index_policy_4]#[((x - min(fairness_index_policy_4)) / (max(fairness_index_policy_4) - min(fairness_index_policy_4))) * (0.347805 - 0.335195) + 0.335195 for x in fairness_index_policy_4]

delay_policy_4 = [8.499761,9.454238,11.628465,15.857472]
local_delay_policy_4 = [7.171397,8.046760,10.031971,14.054391]
offload_delay_policy_4 = [7.709234,7.963700,8.437048,9.174170]
local_queue_length_tasks_policy_4 = [16.411822,18.137248,22.193248,30.594139]
offload_queue_length_tasks_policy_4 = [26.655584,27.512515,28.894733,31.085327]
local_queue_length_bits_policy_4 = [1123.134535,3201.386158,6119.524812,11480.732198]
offload_queue_length_bits_policy_4 = [2111.489564,5434.293109,8626.558891,12471.241802]
offloading_ratios_policy_4 = [0.571330,0.569766,0.569724,0.570725]
battery_energy_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
local_traffic_intensity_constraint_policy_4 = [0.000032,0.037725,0.109635,0.202311]
offload_traffic_intensity_constraint_policy_4 = [0.325476,0.337057,0.354549,0.373842]
local_queue_delay_violation_probability_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_constraint_policy_4 = [0.000000,0.000000,0.000000,0.000000]
rmin_constraint_policy_4 = [0.331696,0.331764,0.331480,0.330589]
local_queue_delay_violation_probability_policy_4 = [0.000000,0.000000,0.000000,0.000000]
offload_queue_delay_violation_probability_policy_4 = [0.000000,0.000000,0.000000,0.000000]
var_reward_policy_4 = [683285579736800.750000,865438622025396.375000,1009628315994037.250000,1380194086358746.500000]
var_energy_policy_4 = [0.000000,0.000000,0.000000,0.000000]
var_throughput_policy_4 = [15992267030741.970703,15964486001365.296875,15984909785664.748047,15883291236487.982422]
var_delay_policy_4 = [4.196698,5.894111,11.260663,29.625467]
var_local_queue_length_bits_policy_4 = [285257.373900,2234178.532525,7733500.521384,26547678.969668]
var_offload_queue_length_bits_policy_4 = [682622.165970,4491435.449731,11331301.209304,24146550.893670]




local_queue_delay_violation_probability_constraint_policy_1 = [0.029253,0.241674,0.338434,0.377138]
offload_queue_delay_violation_probability_constraint_policy_1 = [0.049055,0.400990,0.678218,0.802430]
local_queue_delay_violation_probability_policy_1 = [0.041838,0.117822,0.181660,0.241512]
offload_queue_delay_violation_probability_policy_1 = [0.089012,0.212344,0.316988,0.407439]
local_queue_delay_violation_probability_constraint_policy_2 = [0.031053,0.249775,0.334383,0.346535]
offload_queue_delay_violation_probability_constraint_policy_2 = [0.057606,0.400990,0.665617,0.823132]
local_queue_delay_violation_probability_policy_2 = [0.041573,0.116858,0.185248,0.217112]
offload_queue_delay_violation_probability_policy_2 = [0.093615,0.221118,0.318687,0.412712]
local_queue_delay_violation_probability_constraint_policy_3 = [0.576958,0.900090,0.927993,0.940594]
offload_queue_delay_violation_probability_constraint_policy_3 = [0.026553,0.103510,0.188119,0.245725]
local_queue_delay_violation_probability_policy_3 = [0.175607,0.485306,0.743839,0.848602]
offload_queue_delay_violation_probability_policy_3 = [0.038330,0.077602,0.110870,0.141431]
local_queue_delay_violation_probability_constraint_policy_4 = [0.167867,0.537804,0.624212,0.667417]
offload_queue_delay_violation_probability_constraint_policy_4 = [0.064356,0.392889,0.552655,0.645815]
local_queue_delay_violation_probability_policy_4 = [0.094127,0.255096,0.403572,0.495224]
offload_queue_delay_violation_probability_policy_4 = [0.082720,0.209566,0.289655,0.368062]

reward_policy_1 = [22175244.157802,21373318.222852,20595275.691612,19806380.785671]
reward_policy_2 = [130127346.243434,127755142.688635,122550663.893875,115085303.148416]
reward_policy_3 = [126505491.224009,122481178.900772,117599748.628763,110623191.183135]
# Plotting
# Plotting
#plt.figure(figsize=(15, 8))
#plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')
# Subplot 1: Reward
plt.figure()
# plt.subplot(2, 2, 1)
# plt.plot(task_arrival_rates, reward_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, reward_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, reward_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, reward_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Total System Reward')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.legend(loc="lower left")

#Subplot 2: Energy
#plt.subplot(2, 2, 3)
# plt.plot(task_arrival_rates, energy_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, energy_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, energy_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, energy_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Sum Energy Consumption')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Energy (J)')
# plt.grid(True)
# plt.legend(loc="lower right")

#Subplot 3: Throughput
# plt.subplot(2, 2, 2)
# plt.plot(task_arrival_rates, throughput_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, throughput_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, throughput_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, throughput_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Sum Data Rate')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Throughput (bits/s)')
# plt.grid(True)
# plt.legend(loc="lower right")

# Subplot 4: Fairness Index
# plt.subplot(3, 3, 4)
# plt.plot(task_arrival_rates, fairness_index_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, fairness_index_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, fairness_index_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, fairness_index_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Effect on Fairness Index')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Fairness Index')
# plt.grid(True)
# plt.legend(loc="lower left")

# plt.subplot(3, 3, 4)
# plt.plot(task_arrival_rates, normalized_fairness_index_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, normalized_fairness_index_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, normalized_fairness_index_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, normalized_fairness_index_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Offloading Ratios')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Offloading Ratio')
# plt.grid(True)
# plt.legend(loc="lower left")

#Subplot 5: Delay
# plt.subplot(2, 2, 4)
# plt.plot(task_arrival_rates, delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Sum Delay')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Delay (ms)')
# plt.grid(True)
# plt.legend(loc="upper left")

# # # Subplot 5: Delay
# plt.subplot(3, 3, 6)
# plt.plot(task_arrival_rates, offloading_ratios_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offloading_ratios_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offloading_ratios_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offloading_ratios_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Offloading Ratios')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Offloading Ratio')
# plt.grid(True)
# plt.legend(loc="lower left")



# plt.subplot(3, 3, 7)
# plt.plot(task_arrival_rates, local_delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, local_delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, local_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Sum Local Delay')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Sum Local Delay')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(3, 3, 8)
# plt.plot(task_arrival_rates, offload_delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offload_delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offload_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Sum Offload Delay')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Sum Offload Delay')
# plt.grid(True)
# plt.legend(loc="upper left")


plt.tight_layout()
plt.show()














#plt.figure(figsize=(15, 8))
plt.figure()
#plt.suptitle('Effect of varying Task Arrival Rate on perfomance metrics',fontsize=16, fontweight='bold')

# plt.subplot(2, 2, 1)
# plt.plot(task_arrival_rates, local_queue_length_tasks_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, local_queue_length_tasks_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, local_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_length_tasks_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Local Queue Length (Number of Tasks)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Queue length (Tasks)')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(2, 2, 2)
# plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_length_tasks_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Offload Queue Length (Number of Tasks)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Queue Length (Tasks)')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(2, 2, 3)
# plt.plot(task_arrival_rates, local_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, local_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, local_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Local Queue Length (Number of bits)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Queue Length (bits)')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(2, 2, 4)
# plt.plot(task_arrival_rates, offload_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offload_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offload_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Offload Queue Length (Number of bits)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Queue Length (bits)')
# plt.grid(True)
# plt.legend(loc="upper left")


# plt.subplot(2, 2, 1)
# plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_traffic_intensity_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title(r'Local Traffic Load Constraint Violation Probability ($\Pr \left( \rho_{d,lc}^{(m)} > 1 \right) $)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Violation Probability ')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(2, 2, 2)
# plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_traffic_intensity_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title(r'Offload Traffic Load Constraint Violation Probability ($\Pr \left( \rho_{d,off}^{(m)} > 1 \right) $)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

rmin_constraint_policy_5 = [x - 0.07633 for x in rmin_constraint_policy_1]
# plt.subplot(4, 3, 7)
plt.plot(task_arrival_rates, rmin_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, rmin_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, rmin_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, rmin_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
#plt.plot(task_arrival_rates, rmin_constraint_policy_5, marker='o', color='orange', label=r"$\pi_5$")
plt.title(r'$R^{\min}$ Constraint Violation Probability ($\Pr\left(R_d^{(m)}[t] < R^{\min}\right)$)')
plt.xlabel(r'Task arrival rate ($\lambda$)')
plt.ylabel('Violation Probability')
plt.grid(True)
plt.legend(loc="lower left")


# plt.subplot(2, 2, 3)
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title(r'Local Queue Delay Violation Probability ($\Pr\left( L_{d,lc}^{(m)}[t] > L_{d,lc}^{\max} \right)$)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(2, 2, 4)
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title(r'Offload Queue Delay Violation Probability ($\Pr\left( L_{d,off}^{(m)}[t] > L_{d,off}^{\max} \right)$)')
# plt.xlabel(r'Task arrival rate ($\lambda$)')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 10)
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, local_queue_delay_violation_probability_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Violation Probability (Local Queue Violation Probability Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

# plt.subplot(4, 3, 12)
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_1, marker='o', color='purple', label=r"$\pi_1$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_2, marker='o', color='green', label=r"$\pi_2$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_3, marker='o', color='red', label=r"$\pi_3$")
# plt.plot(task_arrival_rates, offload_queue_delay_violation_probability_constraint_policy_4, marker='o', color='blue', label=r"$\pi_4$")
# plt.title('Violation Probability (Offload Queue Violation Probability Constraint)')
# plt.xlabel('Task arrival rate')
# plt.ylabel('Violation Probability')
# plt.grid(True)
# plt.legend(loc="upper left")

plt.tight_layout()
plt.show()



var_reward_policy_4 = [683285579736800.750000,865438622025396.375000,1009628315994037.250000,1380194086358746.500000]
var_energy_policy_4 = [0.000000,0.000000,0.000000,0.000000]
var_throughput_policy_4 = [15992267030741.970703,15964486001365.296875,15984909785664.748047,15883291236487.982422]
var_delay_policy_4 = [4.196698,5.894111,11.260663,29.625467]
var_local_queue_length_bits_policy_4 = [285257.373900,2234178.532525,7733500.521384,26547678.969668]
var_offload_queue_length_bits_policy_4 = [682622.165970,4491435.449731,11331301.209304,24146550.893670]

plt.figure()
plt.subplot(2, 3, 1)
plt.plot(task_arrival_rates, var_reward_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, var_reward_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, var_reward_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, var_reward_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Reward Variance')
plt.xlabel('Task arrival rate')
plt.grid(True)
plt.legend(loc="lower left")


plt.subplot(2, 3, 2)
plt.plot(task_arrival_rates, var_throughput_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, var_throughput_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, var_throughput_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, var_throughput_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Throughput Variance')
plt.xlabel('Task arrival rate')
plt.grid(True)
plt.legend(loc="lower left")


plt.subplot(2, 3, 3)
plt.plot(task_arrival_rates, var_energy_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, var_energy_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, var_energy_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, var_energy_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Energy Variance')
plt.xlabel('Task arrival rate')
plt.grid(True)
plt.legend(loc="lower left")


plt.subplot(2, 3, 4)
plt.plot(task_arrival_rates, var_delay_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, var_delay_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, var_delay_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, var_delay_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Delay Variance')
plt.xlabel('Task arrival rate')
plt.grid(True)
plt.legend(loc="lower left")



plt.subplot(2, 3, 5)
plt.plot(task_arrival_rates, var_local_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, var_local_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, var_local_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, var_local_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Local Queue length (bits) Variance')
plt.xlabel('Task arrival rate')
plt.grid(True)
plt.legend(loc="lower left")


plt.subplot(2, 3, 6)
plt.plot(task_arrival_rates, var_offload_queue_length_bits_policy_1, marker='o', color='purple', label=r"$\pi_1$")
plt.plot(task_arrival_rates, var_offload_queue_length_bits_policy_2, marker='o', color='green', label=r"$\pi_2$")
plt.plot(task_arrival_rates, var_offload_queue_length_bits_policy_3, marker='o', color='red', label=r"$\pi_3$")
plt.plot(task_arrival_rates, var_offload_queue_length_bits_policy_4, marker='o', color='blue', label=r"$\pi_4$")
plt.title('Offload Queue length (bits) Variance')
plt.xlabel('Task arrival rate')
plt.grid(True)
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()


