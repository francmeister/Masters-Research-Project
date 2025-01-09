import numpy as np
import matplotlib.pyplot as plt
from numpy import interp
##############Model Trained with 10**(-18)###############
##############q_delay = 10**6, contraint q_multipliers = 0############################
# Model trained with q_delay = 10**5 and q_energy = 1.5*10**10
energy_multiplier = [-165,-170,-175,-180,-185]
energy_multiplier = np.log10(energy_multiplier)
reward_values_10_18 = []
energy_values_10_18 = []
throughput_values_10_18 = []
delay_values_10_18 = []
av_local_queue_lengths_bits_10_18 = []
av_offload_queue_lengths_bits_10_18 = []
av_local_queue_lengths_tasks_10_18 = []
av_offload_queue_lengths_tasks_10_18 = []
av_offlaoding_ratios_10_18 = []

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

energy_multiplier = [-185,-180,-175,-170,-165]
energy_multiplier = np.log10(energy_multiplier)
reward_values_10_18 = [25625491.996810,20687022.836980,22189731.328741,11105423.746819,4941179.145194]
energy_values_10_18 = [0.000912,0.000900,0.000907,0.000914,0.000930]
throughput_values_10_18 = [36989717.262070,33104087.759127,34249612.575421,27079792.074444,24559646.360397]
delay_values_10_18 = [22.456026,34.151611,29.859143,68.389737,103.179972]
av_local_queue_lengths_bits_10_18 = [168.6059405940594,144.33321332133212,130.5141314131413,154.24617461746175,179.93294329432945]
av_offload_queue_lengths_bits_10_18 = [5552.757605760576,9940.667146714672,8305.854635463545,22562.380468046802,35078.4702070207]
av_local_queue_lengths_tasks_10_18 = [0.8228622862286229,0.7147614761476148,0.6456345634563457,0.7587758775877589,0.8825382538253824]
av_offload_queue_lengths_tasks_10_18 = [10.718271827182718,19.122142214221423,15.904680468046804,43.1967596759676,66.93069306930694]
av_offlaoding_ratios_10_18 = [0.8808975898683816,0.8816815768356601,0.8825958138941026,0.8789799167222536,0.8792839088524004]

# reward_values_10_18 = [72568171.177887,35194340.481932,-11971097.616234]
# energy_values_10_18 = [0.000904,0.000915,0.000926]
# throughput_values_10_18 = [82763066.535240,45845751.799019,14780968.580456]
# delay_values_10_18 = [11.564663,14.990158,174.882469]
# av_local_queue_lengths_bits_10_18 = [157.3103510351035,167.00855085508553,158.60594059405943]
# av_offload_queue_lengths_bits_10_18 = [1523.5771377137717,2763.9926192619264,65057.32583258325]
# av_local_queue_lengths_tasks_10_18 = [0.7667866786678669,0.8169216921692168,0.7785778577857786]
# av_offload_queue_lengths_tasks_10_18 = [2.915031503150315,5.375157515751575,123.46804680468044]
# av_offlaoding_ratios_10_18 = [0.8804075249915144,0.8771433110005553,0.8772376641353106]

import matplotlib.pyplot as plt
import numpy as np
# Energy constant values (logarithmic scale)
energy_multiplier_log = [-185,-180,-175,-170,-165]
#energy_multiplier_log = [-250,-200,-150]
#energy_multiplier_log = np.log10(energy_multiplier)

# Data for models trained at different constants
metrics = {
    "Reward": [reward_values_10_18],
    "Energy": [energy_values_10_18],
    "Throughput": [throughput_values_10_18],
    "Delay": [delay_values_10_18],
    "Local Queue Length (bits)": [av_local_queue_lengths_bits_10_18],
    "Offload Queue Length (bits)": [av_offload_queue_lengths_bits_10_18],
    "Local Queue Length (tasks)": [av_local_queue_lengths_tasks_10_18],
    "Offload Queue Length (tasks)": [av_offload_queue_lengths_tasks_10_18],
    "Offloading Ratio": [av_offlaoding_ratios_10_18],
}

# Model labels
model_labels = ["Trained at 10^(-18)"]

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
axs = axs.flatten()

for idx, (metric_name, metric_data) in enumerate(metrics.items()):
    for model_idx, model_data in enumerate(metric_data):
        axs[idx].plot(energy_multiplier_log, model_data, label=model_labels[model_idx], marker='o')  # Add circle markers
    axs[idx].set_title(metric_name)
    axs[idx].set_xlabel("Thermal Noise Density(dBm)")
    axs[idx].set_ylabel(metric_name)
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()
