import matplotlib.pyplot as plt

# # Data Models trained with p=0.1
# number_of_urllc_users = [3, 7, 11]
# embb_throughput_values_with_multiplexing_0_1_prob = [23907817.026268, 20052684.695653, 22597040.363259]
# embb_throughput_values_with_multiplexing_0_5_prob = [22130133.288503, 17851856.473591, 18105049.722109]
# embb_throughput_values_with_multiplexing_0_9_prob = [20852953.718298, 17203101.086708, 17015689.744271]

# number_of_embb_users = [3, 7, 11]
# embb_energy_values_with_multiplexing = [0.190020, 0.443501, 0.696766]
# embb_energy_values_without_multiplexing = [0.067788, 0.161190, 0.254607]
# embb_delay_values_with_multiplexing = [258.258311, 606.930736, 951.384825]
# embb_delay_values_without_multiplexing = [3.198654, 7.425685, 11.754907]

# probabilities = [0.1, 0.5, 0.9]
# throughput_3_users = [embb_throughput_values_with_multiplexing_0_1_prob[0], embb_throughput_values_with_multiplexing_0_5_prob[0], embb_throughput_values_with_multiplexing_0_9_prob[0]]
# throughput_7_users = [embb_throughput_values_with_multiplexing_0_1_prob[1], embb_throughput_values_with_multiplexing_0_5_prob[1], embb_throughput_values_with_multiplexing_0_9_prob[1]]
# throughput_11_users = [embb_throughput_values_with_multiplexing_0_1_prob[2], embb_throughput_values_with_multiplexing_0_5_prob[2], embb_throughput_values_with_multiplexing_0_9_prob[2]]

# Data Models trained with p=0.5
number_of_urllc_users = [3, 7, 11]                  
embb_throughput_values_with_multiplexing_0_1_prob = [24437778.331512, 20052684.695653, 22597040.363259]
embb_throughput_values_with_multiplexing_0_5_prob = [22130133.288503, 17851856.473591, 18105049.722109]
embb_throughput_values_with_multiplexing_0_9_prob = [20852953.718298, 17203101.086708, 17015689.744271]

# embb_throughput_values_with_multiplexing_0_1_prob = [31497189.760301, 23396740.413687, 24992584.402866]
# embb_throughput_values_with_multiplexing_0_5_prob = [31497189.760301, 23396740.413687, 24992584.402866]
# embb_throughput_values_with_multiplexing_0_9_prob = [28881229.136653, 23396740.413687, 24992584.402866]

number_of_embb_users = [3, 7, 11]
embb_energy_values_with_multiplexing = [0.190019, 0.443501, 0.696766]
embb_energy_values_without_multiplexing = [0.067788, 0.161190, 0.254607]
embb_delay_values_with_multiplexing = [258.263014, 606.930736, 951.384825]
embb_delay_values_without_multiplexing = [3.198654, 7.425685, 11.754907]
embb_throughput_values_with_multiplexing = [26870817.032178,20149314.265330,19673293.132581]
embb_throughput_values_without_multiplexing = [36181414.966455,31511069.986704,34897126.303756]

probabilities = [0.1, 0.5, 0.9]
throughput_3_users = [embb_throughput_values_with_multiplexing_0_1_prob[0], embb_throughput_values_with_multiplexing_0_5_prob[0], embb_throughput_values_with_multiplexing_0_9_prob[0]]
throughput_7_users = [embb_throughput_values_with_multiplexing_0_1_prob[1], embb_throughput_values_with_multiplexing_0_5_prob[1], embb_throughput_values_with_multiplexing_0_9_prob[1]]
throughput_11_users = [embb_throughput_values_with_multiplexing_0_1_prob[2], embb_throughput_values_with_multiplexing_0_5_prob[2], embb_throughput_values_with_multiplexing_0_9_prob[2]]

# Create a figure with 4 subplots (4 rows, 1 column)
fig, axs = plt.subplots(1, 2)

# Subplot 1: Throughput vs. Number of URLLC Users
axs[0].plot(number_of_urllc_users, embb_throughput_values_with_multiplexing_0_1_prob, marker='o', label='0.1 Prob. URLLC')
axs[0].plot(number_of_urllc_users, embb_throughput_values_with_multiplexing_0_5_prob, marker='o', label='0.5 Prob. URLLC')
axs[0].plot(number_of_urllc_users, embb_throughput_values_with_multiplexing_0_9_prob, marker='o', label='0.9 Prob. URLLC')
axs[0].set_xlabel('Number of URLLC Users')
axs[0].set_ylabel('eMBB Throughput (bps)')
axs[0].set_title('eMBB Throughput vs. Number of URLLC Users')
axs[0].legend()
axs[0].grid(True)

# # Subplot 2: Energy vs. Number of eMBB Users
# axs[1].plot(number_of_embb_users, embb_energy_values_with_multiplexing, marker='o', label='With Multiplexing')
# axs[1].plot(number_of_embb_users, embb_energy_values_without_multiplexing, marker='o', label='Without Multiplexing')
# axs[1].set_xlabel('Number of eMBB Users')
# axs[1].set_ylabel('Energy (J)')
# axs[1].set_title('eMBB Energy vs. Number of eMBB Users')
# axs[1].legend()
# axs[1].grid(True)

# # Subplot 3: Delay vs. Number of eMBB Users
# axs[1,0].plot(number_of_embb_users, embb_delay_values_with_multiplexing, marker='o', label='With Multiplexing')
# axs[1,0].plot(number_of_embb_users, embb_delay_values_without_multiplexing, marker='o', label='Without Multiplexing')
# axs[1,0].set_xlabel('Number of eMBB Users')
# axs[1,0].set_ylabel('Delay (ms)')
# axs[1,0].set_title('eMBB Delay vs. Number of eMBB Users')
# axs[1,0].legend()
# axs[1,0].grid(True)

# Subplot 4: Throughput vs. Probability of Generating Packets
axs[1].plot(probabilities, throughput_3_users, marker='o', label='3 URLLC Users')
axs[1].plot(probabilities, throughput_7_users, marker='o', label='7 URLLC Users')
axs[1].plot(probabilities, throughput_11_users, marker='o', label='11 URLLC Users')
axs[1].set_xlabel('Probability of Generating Packets')
axs[1].set_ylabel('eMBB Throughput (bps)')
axs[1].set_title('eMBB Throughput vs. Probability of Generating Packets')
axs[1].legend()
axs[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2)

# Subplot 2: Energy vs. Number of eMBB Users
axs[0,0].plot(number_of_embb_users, embb_energy_values_with_multiplexing, marker='o', label='With Multiplexing')
axs[0,0].plot(number_of_embb_users, embb_energy_values_without_multiplexing, marker='o', label='Without Multiplexing')
axs[0,0].set_xlabel('Number of eMBB Users')
axs[0,0].set_ylabel('Energy (J)')
axs[0,0].set_title('eMBB Energy vs. Number of eMBB Users')
axs[0,0].legend()
axs[0,0].grid(True)

# Subplot 3: Delay vs. Number of eMBB Users
axs[1,0].plot(number_of_embb_users, embb_delay_values_with_multiplexing, marker='o', label='With Multiplexing')
axs[1,0].plot(number_of_embb_users, embb_delay_values_without_multiplexing, marker='o', label='Without Multiplexing')
axs[1,0].set_xlabel('Number of eMBB Users')
axs[1,0].set_ylabel('Delay (ms)')
axs[1,0].set_title('eMBB Delay vs. Number of eMBB Users')
axs[1,0].legend()
axs[1,0].grid(True)

# Subplot 3: Delay vs. Number of eMBB Users
axs[0,1].plot(number_of_embb_users, embb_throughput_values_with_multiplexing, marker='o', label='With Multiplexing')
axs[0,1].plot(number_of_embb_users, embb_throughput_values_without_multiplexing, marker='o', label='Without Multiplexing')
axs[0,1].set_xlabel('Number of eMBB Users')
axs[0,1].set_ylabel('Sum Throughput (bps)')
axs[0,1].set_title('eMBB Sum Throughput vs. Number of eMBB Users')
axs[0,1].legend()
axs[0,1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# Analysis and Deductions
# 1. eMBB Throughput vs. Number of URLLC Users:
#    - Throughput decreases as the number of URLLC users increases for all probabilities.
#    - Higher probabilities (e.g., 0.9) lead to lower throughput due to increased interference from URLLC.
#
# 2. eMBB Energy vs. Number of eMBB Users:
#    - Energy consumption increases with the number of eMBB users.
#    - Multiplexing consumes more energy compared to non-multiplexing.
#
# 3. eMBB Delay vs. Number of eMBB Users:
#    - Delay increases with the number of eMBB users.
#    - Multiplexing results in significantly higher delay compared to non-multiplexing.
#
# 4. eMBB Throughput vs. Probability of Generating Packets:
#    - Throughput decreases as the probability of generating packets increases.
#    - Larger numbers of URLLC users result in a more pronounced decrease in throughput.

# number_of_urllc_users = [3, 7, 11]   
# embb_throughput_values_with_multiplexing_0_1_prob = [31497189.760301, 23396740.413687, 24992584.402866]
# embb_throughput_values_with_multiplexing_0_5_prob = [31497189.760301, 23396740.413687, 22269987.023548]
# embb_throughput_values_with_multiplexing_0_9_prob = [28881229.136653, 23396740.413687, 24992584.402866]

# outage_probabilities_0_1 = []
# outage_probabilities_0_5 = []
# outage_probabilities_0_9 = [,,0.563455]

number_of_urllc_users = [3, 7, 11]                  23324014.348767
embb_throughput_values_with_multiplexing_0_1_prob = [26376640.654661, 21027447.105985, 22597040.363259]
embb_throughput_values_with_multiplexing_0_5_prob = [24982482.129859, 17851856.473591, 18105049.722109]
embb_throughput_values_with_multiplexing_0_9_prob = [24424910.124819, 17203101.086708, 17015689.744271]

embb_outage_prob_values_with_multiplexing_0_1_prob = [0.035644, 0.015095, 22597040.363259]
embb_outage_prob_values_with_multiplexing_0_5_prob = [0.143812, 17851856.473591, 18105049.722109]
embb_outage_prob_values_with_multiplexing_0_9_prob = [0.514256, 17203101.086708, 17015689.744271]