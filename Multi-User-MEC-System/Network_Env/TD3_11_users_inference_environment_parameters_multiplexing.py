import numpy as np
import matplotlib.pyplot as plt

# Effecet of changing task arrival rate on reward, energy, throughput and fairness
import matplotlib.pyplot as plt

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

reward_policy_3_multiplexing = []
energy_policy_3_multiplexing = []
throughput_policy_3_multiplexing = []
fairness_index_policy_3_multiplexing = []
delay_policy_3_multiplexing = []

local_delay_policy_3_multiplexing = []
offload_delay_policy_3_multiplexing = []
local_queue_length_tasks_policy_3_multiplexing = []
offload_queue_length_tasks_policy_3_multiplexing = []
local_queue_length_bits_policy_3_multiplexing = []
offload_queue_length_bits_policy_3_multiplexing = []

offloading_ratios_policy_3_multiplexing = []
battery_energy_constraint_policy_3_multiplexing = [0.0,0.0,0.0,0.0,0.0]
local_traffic_intensity_constraint_policy_3_multiplexing = []
offload_traffic_intensity_constraint_policy_3_multiplexing = []
local_queue_delay_violation_probability_constraint_policy_3_multiplexing = []
offload_queue_delay_violation_probability_constraint_policy_3_multiplexing = []
rmin_constraint_policy_3_multiplexing = []
local_queue_delay_violation_probability_policy_3_multiplexing = []
offload_queue_delay_violation_probability_policy_3_multiplexing = []