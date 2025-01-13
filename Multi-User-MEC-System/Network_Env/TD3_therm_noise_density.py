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
reward_values_10_18 = [26136211.342611,20327072.468967,16811535.727283,11543723.961690,7635109.484065]
energy_values_10_18 = [0.000912,0.000916,0.000906,0.000904,0.000906]
throughput_values_10_18 = [41608332.317442,37147025.502852,34199509.738155,29753766.955433,27805733.716985]
delay_values_10_18 = [17.974603,30.855394,37.909046,46.511277,65.801212]
av_local_queue_lengths_bits_10_18 = [152.23591359135915,168.0136813681368,162.67551755175515,164.10153015301532,159.58199819982]
av_offload_queue_lengths_bits_10_18 = [4071.8451845184522,8577.184698469848,11072.825742574256,14316.306840684067,21236.81305130513]
av_local_queue_lengths_tasks_10_18 = [0.7506750675067507,0.817011701170117,0.7704770477047705,0.7998199819981998,0.7576957695769576]
av_offload_queue_lengths_tasks_10_18 = [7.8511251125112524,16.530333033303332,21.13969396939694,27.424032403240325,40.28910891089109]
av_offlaoding_ratios_10_18 = [0.8784204622571551,0.8759411722985644,0.8795477990415741,0.877208979371154,0.8795472888861574]

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
