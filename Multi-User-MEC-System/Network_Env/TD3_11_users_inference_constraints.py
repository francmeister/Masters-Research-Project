import numpy as np
import matplotlib.pyplot as plt

## Rmin constraint
# Data preparation
constraint_multiplier = [10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9)]
constraint_multiplier = np.log10(constraint_multiplier)

reward_values = [-26369825.470007,-25775695.420629,-17596814.615557,-32157607.129300,-36774903.366136,-99477743.804507,-645392539.892048,-5997799164.403359,-59827379885.043037]
energy_values = [0.000958,0.000963,0.000961,0.000959,0.000961,0.000954,0.000983,0.000954,0.000956]
throughput_values = [26940696.609926,29682922.969028,32972285.712979,27639199.839450,28113129.061962,25795880.416850,29559599.562923,33188294.067138,33569733.086985]
fairness_index = [0.528562,0.532791,0.520348,0.537520,0.519874,0.527377,0.528455,0.538142,0.540710]
delay_values = [77.888891,82.020111,72.178764,89.632650,88.875271,102.627383,86.434636,71.135701,66.131201]
offloading_ratios = [0.7204445947659744,0.7127795474683686,0.7109744302605455,0.7137283346442621,0.7102113356888459,0.7004136728569706,0.699307153735838,0.7203668943785957,0.7180457839421677]
constraint_violation_count = [3.6702970297029704,3.607920792079208,3.5782178217821783,3.5702970297029704,3.607920792079208,3.599009900990099,3.6613861386138615,3.5564356435643565,3.5564356435643565]

# reward_values = [-6737097.562934,-8197872.661078,-9123563.886012,-8969997.767659,-14026483.460241,-69185741.227507,-615562485.544707,-6006635458.792838,-60030546138.336044]
# energy_values = [0.000914,0.000915,0.000920,0.000904,0.000911,0.000918,0.000925,0.000903,0.000912]
# throughput_values = [32406595.099494,32636853.828215,32563341.137231,32366006.421213,32651535.639765,32637616.702072,32359014.932564,32655553.507831,32662152.820080]
# fairness_index = [0.497544,0.494824,0.494043,0.498483,0.495510,0.490566,0.497615,0.494906,0.498993]
# delay_values = [50.861772,54.211180,55.651475,54.352314,54.038446,53.166910,53.959949,54.810001,53.391962]
# offloading_ratios = [0.8734831108081124,0.8738177676792556,0.8738386453108595,0.8736967517047767,0.8732927474480092,0.8744910888217322,0.8743519520931989,0.8753047826556116,0.8713230651229362]
# constraint_violation_count = [3.607920792079208,3.5643564356435644,3.589108910891089,3.59009900990099,3.5801980198019803,3.6564356435643566,3.61980198019802,3.5821782178217823,3.5811881188118813]

# Plotting
plt.figure(figsize=(10, 7))
#plt.suptitle("Metrics vs Constraint Multiplier", fontsize=16, fontweight='bold')
plt.suptitle(r"$R_d^{(m)}[t] \geq R^{min}, \, \forall d \in D^{(m)}$", 
             fontsize=16, fontweight='bold')

# plt.suptitle(r"Constraint (3.32): $H_d^{(m)}[t-1] + \omega_d^{(m)}[t] > E_d^{(m)}[t], \, \forall d \in D^{(m)}$", 
#              fontsize=16, fontweight='bold')

# Reward values
plt.subplot(3, 3, 1)
plt.plot(constraint_multiplier, reward_values, marker='o', label="Reward")
plt.title("Reward vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Reward Values")
plt.grid(True)
plt.legend()

# Energy values
plt.subplot(3, 3, 2)
plt.plot(constraint_multiplier, energy_values, marker='o', color='orange', label="Energy")
plt.title("Energy vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Energy Values")
plt.grid(True)
plt.legend()

# Throughput values
plt.subplot(3, 3, 3)
plt.plot(constraint_multiplier, throughput_values, marker='o', color='green', label="Throughput")
plt.title("Throughput vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Throughput Values")
plt.grid(True)
plt.legend()

# Fairness index
plt.subplot(3, 3, 4)
plt.plot(constraint_multiplier, fairness_index, marker='o', color='red', label="Fairness Index")
plt.title("Fairness Index vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Fairness Index")
plt.grid(True)
plt.legend()

# Delay values
plt.subplot(3, 3, 5)
plt.plot(constraint_multiplier, delay_values, marker='o', color='purple', label="Delay")
plt.title("Delay vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Delay Values")
plt.grid(True)
plt.legend()

# Offloading ratios
plt.subplot(3, 3, 6)
plt.plot(constraint_multiplier, offloading_ratios, marker='o', color='brown', label="Offloading Ratio")
plt.title("Offloading Ratio vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Offloading Ratios")
plt.grid(True)
plt.legend()

plt.subplot(3, 3, 7)
plt.plot(constraint_multiplier, constraint_violation_count, marker='o', color='blue', label="Constraint Violation Count")
plt.title("Constraint Violation Count vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Constraint Violation Count")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #battery energy constraint

constraint_multiplier = [10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9)]
constraint_multiplier = np.log10(constraint_multiplier)

reward_values = [-22601241.624703,-27766700.390478,-22097687.014003,-26387138.136698,-25624803.649431,-31848547.916832,84639413.343214,1067123063.285585,10986573736.410425]
energy_values = [0.000956,0.000954,0.000963,0.000956,0.000978,0.000957,0.000962,0.000972,0.000952]
throughput_values = [29304836.517472,29408360.753738,30199992.246349,30770253.087116,30214456.593700,23334669.629564,30263902.196472,28913580.598421,34244309.451468]
fairness_index = [0.540254,0.522607,0.533424,0.503044,0.523246,0.528095,0.528942,0.513051,0.529455]
delay_values = [75.145816,85.733762,75.726116,85.857457,84.535547,103.654106,82.403623,94.419858,66.789672]
offloading_ratios = [0.7212751111941341,0.7021343274698932,0.7171364163303472,0.7140999615741392,0.701279571335168,0.7051011697934129,0.7069149515016272,0.7111821190493971,0.7230811730028772]
constraint_violation_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# reward_values = [6879765.611202,6094630.710882,7005604.731830,6211317.935425,6868485.017851,15797352.979709,116568943.471356,1106131177.021166,11005926269.563887]
# energy_values = [0.000917,0.000910,0.000910,0.000907,0.000915,0.000911,0.000923,0.000926,0.000912]
# throughput_values = [35125509.967335,34772844.267331,35001960.409995,35150160.733143,34957433.660847,34979301.454541,34870568.455480,34935819.180719,34999478.601413]
# fairness_index = [0.537759,0.533804,0.532565,0.543197,0.534978,0.530819,0.532202,0.534956,0.532975]
# delay_values = [28.982217,30.063582,28.704118,30.888015,30.942257,33.027062,28.906226,29.816632,30.779746]
# offloading_ratios = [0.873561340248364,0.8772086977110642,0.8747104956388377,0.8751970694621355,0.8719011398100427,0.8729715346535127,0.8529252925292531,0.8711738840021397,0.8745932861859447]
# constraint_violation_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
# Plotting
plt.figure(figsize=(10, 7))
#plt.suptitle("Metrics vs Constraint Multiplier", fontsize=16, fontweight='bold')
plt.suptitle(r"Constraint (3.32): $H_d^{(m)}[t-1] + \omega_d^{(m)}[t] > E_d^{(m)}[t], \, \forall d \in D^{(m)}$", 
             fontsize=16, fontweight='bold')

# Reward values
plt.subplot(3, 2, 1)
plt.plot(constraint_multiplier, reward_values, marker='o', label="Reward")
plt.title("Reward vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Reward Values")
plt.grid(True)
plt.legend()

# Energy values
plt.subplot(3, 2, 2)
plt.plot(constraint_multiplier, energy_values, marker='o', color='orange', label="Energy")
plt.title("Energy vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Energy Values")
plt.grid(True)
plt.legend()

# Throughput values
plt.subplot(3, 2, 3)
plt.plot(constraint_multiplier, throughput_values, marker='o', color='green', label="Throughput")
plt.title("Throughput vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Throughput Values")
plt.grid(True)
plt.legend()

# # Fairness index
# plt.subplot(3, 3, 4)
# plt.plot(constraint_multiplier, fairness_index, marker='o', color='red', label="Fairness Index")
# plt.title("Fairness Index vs Constraint Multiplier")
# plt.xlabel("Log10(Constraint Multiplier)")
# plt.ylabel("Fairness Index")
# plt.grid(True)
# plt.legend()

# Delay values
plt.subplot(3, 2, 4)
plt.plot(constraint_multiplier, delay_values, marker='o', color='purple', label="Delay")
plt.title("Delay vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Delay Values")
plt.grid(True)
plt.legend()

# Offloading ratios
plt.subplot(3, 2, 5)
plt.plot(constraint_multiplier, offloading_ratios, marker='o', color='brown', label="Offloading Ratio")
plt.title("Offloading Ratio vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Offloading Ratios")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(constraint_multiplier, constraint_violation_count, marker='o', color='blue', label="Constraint Violation Count")
plt.title("Constraint Violation Count vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Constraint Violation Count")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #offload traffic intensity constraint

constraint_multiplier = [10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9)]
constraint_multiplier = np.log10(constraint_multiplier)

reward_values = [-17687350.913985,-23421997.096891,-16686384.564899,-22061014.674176,-21726561.509824,-9188340.840865,-3910978.660952,134398751.113306,1734692946.386289]
energy_values = [0.000959,0.000963,0.000963,0.000963,0.000979,0.000947,0.000965,0.000971,0.000970]
throughput_values = [32514407.154953,31982916.270142,32425255.922430,30640579.189195,31509292.969909,33844975.087256,30693030.389169,27030757.855462,30386272.004777]
fairness_index = [0.525758,0.542942,0.533438,0.527570,0.534815,0.544528,0.528915,0.533803,0.534452]
delay_values = [71.645800,81.923354,69.336596,76.541901,77.531364,62.938282,79.007280,88.530432,69.376574]
offloading_ratios = [0.7230591695060073,0.6994923031263752,0.7251942247526371,0.7178750220089615,0.7073003833679337,0.7229708050964035,0.706229914906716,0.699281051673537,0.7151718069346034]
constraint_violation_count = [4.57029702970297,4.401980198019802,4.583168316831683,4.703960396039604,4.55049504950495,4.371287128712871,4.646534653465347,4.840594059405941,4.654455445544555]

# reward_values = [-17467405.298741,-17908426.491187,-19426853.705670,-18891876.387245,-19508368.047760,-17469432.935845,-22780834.228947,-30445342.781154,-248144520.201619]
# energy_values = [0.000918,0.000919,0.000910,0.000909,0.000910,0.000916,0.000907,0.000905,0.000929]
# throughput_values = [26473589.174698,26550685.900123,26682276.338606,26446609.803605,26438537.381639,26605791.914143,26536989.149512,26498614.494915,26438535.244550]
# fairness_index = [0.539196,0.536624,0.535256,0.537296,0.534084,0.540979,0.533764,0.539859,0.532255]
# delay_values = [60.333915,61.362116,64.907861,63.397811,64.532052,60.557835,65.017973,60.065202,60.553085]
# offloading_ratios = [0.8705973852614431,0.8728804506296473,0.8726227345223958,0.8716612472688425,0.8713040382372929,0.873722630356086,0.8760742924626004,0.87478848769749,0.87314046221716]
# constraint_violation_count = [5.792079207920792,5.846534653465347,5.821782178217822,5.824752475247525,5.842574257425743,5.799009900990099,5.828712871287129,5.834653465346535,5.8396039603960395]

# Plotting
plt.figure(figsize=(10, 7))
#plt.suptitle("Metrics vs Constraint Multiplier", fontsize=16, fontweight='bold')
plt.suptitle(r"Constraint (3.34): $\rho_{d,{off}}^{(m)} \leq 1, \, \forall d \in D^{(m)}$", 
             fontsize=16, fontweight='bold')

# Reward values
plt.subplot(3, 2, 1)
plt.plot(constraint_multiplier, reward_values, marker='o', label="Reward")
plt.title("Reward vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Reward Values")
plt.grid(True)
plt.legend()

# Energy values
plt.subplot(3, 2, 2)
plt.plot(constraint_multiplier, energy_values, marker='o', color='orange', label="Energy")
plt.title("Energy vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Energy Values")
plt.grid(True)
plt.legend()

# Throughput values
plt.subplot(3, 2, 3)
plt.plot(constraint_multiplier, throughput_values, marker='o', color='green', label="Throughput")
plt.title("Throughput vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Throughput Values")
plt.grid(True)
plt.legend()

# # Fairness index
# plt.subplot(3, 3, 4)
# plt.plot(constraint_multiplier, fairness_index, marker='o', color='red', label="Fairness Index")
# plt.title("Fairness Index vs Constraint Multiplier")
# plt.xlabel("Log10(Constraint Multiplier)")
# plt.ylabel("Fairness Index")
# plt.grid(True)
# plt.legend()

# Delay values
plt.subplot(3, 2, 4)
plt.plot(constraint_multiplier, delay_values, marker='o', color='purple', label="Delay")
plt.title("Delay vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Delay Values")
plt.grid(True)
plt.legend()

# Offloading ratios
plt.subplot(3, 2, 5)
plt.plot(constraint_multiplier, offloading_ratios, marker='o', color='brown', label="Offloading Ratio")
plt.title("Offloading Ratio vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Offloading Ratios")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(constraint_multiplier, constraint_violation_count, marker='o', color='blue', label="Constraint Violation Count")
plt.title("Constraint Violation Count vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Constraint Violation Count")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Local traffic intensity constraint

constraint_multiplier = [10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9)]
constraint_multiplier = np.log10(constraint_multiplier)

reward_values = [-25761088.350324,-34632711.594853,-28020058.685729,-19192237.742621,-23669501.789214,-16735338.707979,24200786.915765,520923322.876482,5312624235.271771]
energy_values = [0.000957,0.000954,0.000958,0.000969,0.000953,0.000966,0.000968,0.000968,0.000965]
throughput_values = [28765589.922075,26270530.656149,27663799.102812,32091637.863525,28654539.074852,31258478.564241,27431813.392915,32466440.424080,32931489.849273]
fairness_index = [0.522209,0.526913,0.528063,0.536333,0.532739,0.526185,0.525095,0.525878,0.532082]
delay_values = [80.356242,93.194486,82.632891,73.615489,77.146762,77.383787,85.223464,75.581299,80.245796]
offloading_ratios = [0.7174769432101784,0.7115668229877042,0.711908578816873,0.7116033458621795,0.7089816877900388,0.7181763709060828,0.6958924202986494,0.7051039547971747,0.7139327860983884]
constraint_violation_count = [1.6821782178217821,1.7900990099009901,1.706930693069307,1.7405940594059406,1.6594059405940593,1.6613861386138613,1.8514851485148516,1.7633663366336634,1.6603960396039603]

# reward_values = [-6799964.006207,-6701971.936313,-6785666.116450,-5963032.492845,-6455312.071469,-4260096.521114,5290508.386057,114704608.034731,1392459874.358421]
# energy_values = [0.000910,0.000905,0.000918,0.000910,0.000917,0.000905,0.000921,0.000919,0.000916]
# throughput_values = [31567553.348337,31633911.275488,31702556.265963,31494423.131556,31544479.745213,31665048.719604,31738374.694946,31749741.473369,31504190.358866]
# fairness_index = [0.523666,0.527829,0.520923,0.521365,0.521322,0.520209,0.519792,0.516011,0.523896]
# delay_values = [49.433842,49.511861,49.429769,47.633285,48.802586,46.755488,49.053371,50.626226,51.338730]
# offloading_ratios = [0.8732518306898925,0.8733170986581925,0.8716215652737648,0.8736114070082905,0.8687515690023561,0.87317255128935,0.872466153582605,0.8726542157041609,0.8719956750858223]
# constraint_violation_count = [0.43564356435643564,0.4267326732673267,0.45445544554455447,0.4178217821782178,0.4495049504950495,0.41485148514851483,0.4524752475247525,0.4316831683168317,0.4306930693069307]

# Plotting
plt.figure(figsize=(10, 7))
#plt.suptitle("Metrics vs Constraint Multiplier", fontsize=16, fontweight='bold')
plt.suptitle(r"Constraint (3.33): $\rho_{d,{lc}}^{(m)} \leq 1, \, \forall d \in D^{(m)}$", 
             fontsize=16, fontweight='bold')

# Reward values
plt.subplot(3, 2, 1)
plt.plot(constraint_multiplier, reward_values, marker='o', label="Reward")
plt.title("Reward vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Reward Values")
plt.grid(True)
plt.legend()

# Energy values
plt.subplot(3, 2, 2)
plt.plot(constraint_multiplier, energy_values, marker='o', color='orange', label="Energy")
plt.title("Energy vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Energy Values")
plt.grid(True)
plt.legend()

# Throughput values
plt.subplot(3, 2, 3)
plt.plot(constraint_multiplier, throughput_values, marker='o', color='green', label="Throughput")
plt.title("Throughput vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Throughput Values")
plt.grid(True)
plt.legend()

# Fairness index
# plt.subplot(3, 3, 4)
# plt.plot(constraint_multiplier, fairness_index, marker='o', color='red', label="Fairness Index")
# plt.title("Fairness Index vs Constraint Multiplier")
# plt.xlabel("Log10(Constraint Multiplier)")
# plt.ylabel("Fairness Index")
# plt.grid(True)
# plt.legend()

# Delay values
plt.subplot(3, 2, 4)
plt.plot(constraint_multiplier, delay_values, marker='o', color='purple', label="Delay")
plt.title("Delay vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Delay Values")
plt.grid(True)
plt.legend()

# Offloading ratios
plt.subplot(3, 2, 5)
plt.plot(constraint_multiplier, offloading_ratios, marker='o', color='brown', label="Offloading Ratio")
plt.title("Offloading Ratio vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Offloading Ratios")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(constraint_multiplier, constraint_violation_count, marker='o', color='blue', label="Constraint Violation Count")
plt.title("Constraint Violation Count vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Constraint Violation Count")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #Local queueing violation probability constraint

constraint_multiplier = [10**(1), 10**(2), 10**(3), 10**(4), 10**(5), 10**(6), 10**(7), 10**(8), 10**(9)]
constraint_multiplier = np.log10(constraint_multiplier)

reward_values = [-28455459.229110,-25461045.377218,-17576834.510875,-33187003.220869,-39222244.798157,-11405058.908813,50558284.123412,716089425.741352,7821396939.759587]
energy_values = [0.000967,0.000979,0.000967,0.000951,0.000962,0.000965,0.000963,0.000961,0.000961]
throughput_values = [30323039.971083,29420433.643062,31890739.835180,26761153.025173,25538718.771806,30472356.584225,28054453.334173,27550196.331869,35653856.137962]
fairness_index = [0.515581,0.526645,0.530112,0.533297,0.519049,0.535687,0.535790,0.530200,0.539135]
delay_values = [88.540350,80.379736,69.933421,91.527993,102.120073,70.250283,79.261331,83.156654,63.541054]
offloading_ratios = [0.7016502400055759,0.7042952705188026,0.7134992801733904,0.7049107477229137,0.6947522367682324,0.7181527513467123,0.715674139658263,0.7041588210797838,0.7261053646234429]
constraint_violation_count = [2.517821782178218,2.4544554455445544,2.3336633663366335,2.4425742574257425,2.593069306930693,2.29009900990099,2.3198019801980196,2.4673267326732673,2.213861386138614]
# Plotting
plt.figure(figsize=(10, 7))
#plt.suptitle("Metrics vs Constraint Multiplier", fontsize=16, fontweight='bold')
plt.suptitle(r"Constraint (3.35): $\Pr(L_d^{(m)}[t] > L_d^*) < \epsilon^{{max}}, \, \forall d \in D^{(m)}$", 
             fontsize=16, fontweight='bold')

# Reward values
plt.subplot(3, 2, 1)
plt.plot(constraint_multiplier, reward_values, marker='o', label="Reward")
plt.title("Reward vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Reward Values")
plt.grid(True)
plt.legend()

# Energy values
plt.subplot(3, 2, 2)
plt.plot(constraint_multiplier, energy_values, marker='o', color='orange', label="Energy")
plt.title("Energy vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Energy Values")
plt.grid(True)
plt.legend()

# Throughput values
plt.subplot(3, 2, 3)
plt.plot(constraint_multiplier, throughput_values, marker='o', color='green', label="Throughput")
plt.title("Throughput vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Throughput Values")
plt.grid(True)
plt.legend()

# # Fairness index
# plt.subplot(3, 3, 4)
# plt.plot(constraint_multiplier, fairness_index, marker='o', color='red', label="Fairness Index")
# plt.title("Fairness Index vs Constraint Multiplier")
# plt.xlabel("Log10(Constraint Multiplier)")
# plt.ylabel("Fairness Index")
# plt.grid(True)
# plt.legend()

# Delay values
plt.subplot(3, 2, 4)
plt.plot(constraint_multiplier, delay_values, marker='o', color='purple', label="Delay")
plt.title("Delay vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Delay Values")
plt.grid(True)
plt.legend()

# Offloading ratios
plt.subplot(3, 2, 5)
plt.plot(constraint_multiplier, offloading_ratios, marker='o', color='brown', label="Offloading Ratio")
plt.title("Offloading Ratio vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Offloading Ratios")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(constraint_multiplier, constraint_violation_count, marker='o', color='blue', label="Constraint Violation Count")
plt.title("Constraint Violation Count vs Constraint Multiplier")
plt.xlabel("Log10(Constraint Multiplier)")
plt.ylabel("Constraint Violation Count")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


