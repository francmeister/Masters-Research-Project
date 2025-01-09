import matplotlib.pyplot as plt
import numpy as np

# Data
rmin_multiplier = [10**(2),10**(3),10**(4),10**(5), 10**(6), 10**(7), 10**(8), 10**(9),10**(10)]
rmin_multiplier = np.log10(rmin_multiplier)
reward_values = [17464652.495872,21126276.078622,22365171.972484,16151155.424726, -48437191.371916, -658924866.740888, -6729809178.350162, -68661725249.869659,-672602431256.400635]
energy_values = [0.000905,0.000911,0.000909,0.000917, 0.000916, 0.000924, 0.000905, 0.000930,0.000913]
throughput_values = [30486131.873323,33970431.915864,35135706.972873,34769858.774897, 31537957.109268, 33741456.945204, 32822734.211469, 28620719.007237,34810271.256709]
fairness_index = [0.542636,0.526525,0.539389,0.541992, 0.541348, 0.541535, 0.541276, 0.529709,0.523851]
delay_values = [39.647713,36.695230,30.064255,27.088522, 36.023432, 28.149049, 34.090861, 52.555796,41.484332]


reward_values = [24730484.710708,13402950.882700,17560821.884017,14780509.217297,-53588367.532716,-655127707.322814,-6810932099.747789,-66975077913.456772,-678362105926.008545]
energy_values = [0.000914,0.000919,0.000908,0.000902,0.000914,0.000924,0.000910,0.000907,0.000909]
throughput_values = [36359368.790668,28209412.660894,31543378.505315,33800599.339352,29467862.161358,32599560.879286,35678986.984784,33498246.153457,36079513.959433]
fairness_index = [0.542434,0.530336,0.528377,0.536117,0.519910,0.535951,0.522624,0.540928,0.532258]
delay_values = [24.832326,55.514405,42.243132,31.661157,53.373138,36.688716,31.129756,30.629163,27.124251]
# Create 2x3 subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot each variable
axs[0, 0].plot(rmin_multiplier, reward_values, marker='o', label='Reward', color='blue')
axs[0, 0].set_title('Reward')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].legend()

axs[0, 1].plot(rmin_multiplier, energy_values, marker='o', label='Energy', color='green')
axs[0, 1].set_title('Energy')
axs[0, 1].set_ylabel('Energy')
axs[0, 1].legend()

axs[0, 2].plot(rmin_multiplier, throughput_values, marker='o', label='Throughput', color='orange')
axs[0, 2].set_title('Throughput')
axs[0, 2].set_ylabel('Throughput')
axs[0, 2].legend()

axs[1, 0].plot(rmin_multiplier, fairness_index, marker='o', label='Fairness Index', color='purple')
axs[1, 0].set_title('Fairness Index')
axs[1, 0].set_ylabel('Fairness Index')
axs[1, 0].legend()

axs[1, 1].plot(rmin_multiplier, delay_values, marker='o', label='Delay', color='red')
axs[1, 1].set_title('Delay')
axs[1, 1].set_ylabel('Delay')
axs[1, 1].legend()

# Optional: Leave the last subplot empty or create a combined plot
axs[1, 2].axis('off')  # Remove the axis from the empty subplot
# OR Uncomment below to create a combined plot in the last slot
# axs[1, 2].plot(rmin_multiplier, reward_values, marker='o', label='Reward', color='blue')
# axs[1, 2].plot(rmin_multiplier, energy_values, marker='o', label='Energy', color='green')
# axs[1, 2].plot(rmin_multiplier, throughput_values, marker='o', label='Throughput', color='orange')
# axs[1, 2].plot(rmin_multiplier, fairness_index, marker='o', label='Fairness Index', color='purple')
# axs[1, 2].plot(rmin_multiplier, delay_values, marker='o', label='Delay', color='red')
# axs[1, 2].set_title('Combined')
# axs[1, 2].legend()

# Adjust layout
plt.tight_layout()
plt.show()
