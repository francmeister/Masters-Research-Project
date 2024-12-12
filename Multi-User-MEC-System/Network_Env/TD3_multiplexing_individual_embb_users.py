import matplotlib.pyplot as plt
import numpy as np
# Fix number of urllc users to 8
# 3 embb users
embb_users_3_users = ['eMBB User 1', 'eMBB User 2', 'eMBB User 3']
users_data_rates_3_embb_users = [11688413.44953297,  8874908.6726139,   7000377.51247037] #[embb user 1, embb user 2, embb user 3]
number_of_puncturing_users_3_embb_users = [5, 1, 2]#[embb user 1, embb user 2, embb user 3]
number_of_allocated_RBs_3_embb_users = [7, 4, 1]#[embb user 1, embb user 2, embb user 3]
number_of_clustered_urllc_users_3_embb_users = [5, 1, 2]#[embb user 1, embb user 2, embb user 3]
number_of_failed_urllc_transmissions_3_embb_users = sum(number_of_clustered_urllc_users_3_embb_users) - sum(number_of_puncturing_users_3_embb_users)

# Separate bar plots for each metric
fig, axes = plt.subplots(2, 2)

axes = axes.flatten()
# Data Rate Plot
axes[0].bar(embb_users_3_users, users_data_rates_3_embb_users, color='blue')
axes[0].set_ylabel('Data Rate (bps)')
axes[0].set_title('Data Rate per eMBB User')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Puncturing Users Plot
axes[1].bar(embb_users_3_users, number_of_puncturing_users_3_embb_users, color='orange')
axes[1].set_ylabel('Number of Transmissions')
axes[1].set_title('Number of Transmitting URLLC users per eMBB User')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Allocated RBs Plot
axes[2].bar(embb_users_3_users, number_of_allocated_RBs_3_embb_users, color='green')
axes[2].set_ylabel('Number of Allocated RBs')
axes[2].set_title('Number of Allocated RBs per eMBB User')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

# Clustered URLLC Users Plot
axes[3].bar(embb_users_3_users, number_of_clustered_urllc_users_3_embb_users, color='red')
axes[3].set_ylabel('Number of Clustered URLLC Users')
axes[3].set_title('Number of Clustered URLLC Users per eMBB User')
#axes[1,1].set_xlabel('eMBB Users')
axes[3].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# 7 embb users
embb_users_7_users = ['1', '2', '3','4', '5', '6','7']
users_data_rates_7_embb_users = [818498.25087194, 1893152.68516387,  888877.36992894, 0, 6433690.87033959, 5487796.26671838, 4390733.40909334]
number_of_puncturing_users_7_embb_users = [0, 0, 2, 0, 0, 3, 0]
number_of_allocated_RBs_7_embb_users = [1, 2, 2, 0, 2, 2, 3]
number_of_clustered_urllc_users_7_embb_users = [0, 0, 2, 3, 0, 3, 0]
number_of_failed_urllc_transmissions_7_embb_users = sum(number_of_clustered_urllc_users_7_embb_users) - sum(number_of_puncturing_users_7_embb_users)

# Separate bar plots for each metric
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
# Data Rate Plot
axes[0].bar(embb_users_7_users, users_data_rates_7_embb_users, color='blue')
axes[0].set_ylabel('Data Rate (bps)')
axes[0].set_xlabel('eMBB User Index')
axes[0].set_title('Data Rate per eMBB User')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Puncturing Users Plot
axes[1].bar(embb_users_7_users, number_of_puncturing_users_7_embb_users, color='orange')
axes[1].set_ylabel('Number of Transmissions')
axes[1].set_title('Number of Transmitting URLLC users per eMBB User')
axes[1].set_xlabel('eMBB User Index')
#axes[1].set_title('Puncturing Users per eMBB User')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Allocated RBs Plot
axes[2].bar(embb_users_7_users, number_of_allocated_RBs_7_embb_users, color='green')
axes[2].set_ylabel('Number of Allocated RBs')
axes[2].set_title('Number of Allocated RBs per eMBB User')
axes[2].set_xlabel('eMBB User Index')
#axes[2].set_title('Allocated RBs per eMBB User')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

# Clustered URLLC Users Plot
axes[3].bar(embb_users_7_users, number_of_clustered_urllc_users_7_embb_users, color='red')
axes[3].set_ylabel('Number of Clustered URLLC Users')
axes[3].set_title('Number of Clustered URLLC Users per eMBB User')
axes[3].set_xlabel('eMBB User Index')
#axes[3].set_title('Clustered URLLC Users per eMBB User')
#axes[1,1].set_xlabel('eMBB Users')
axes[3].grid(axis='y', linestyle='--', alpha=0.7)

# # Rotate x-axis labels
# for ax in axes:
#     ax.set_xticklabels(embb_users_7_users, rotation=45, ha='right')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# 11 embb users
embb_users_11_users = ['1', '2', '3','4', '5', '6','7','8', '9', '10','11']
users_data_rates_11_embb_users = [2479048.14151504,0,0,2777366.21315914,4140749.62301344, 4887950.12625243, 2392151.59271831, 2328116.48254063,4980973.86333514,0,0]
number_of_puncturing_users_11_embb_users = [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
number_of_allocated_RBs_11_embb_users = [1, 0, 0, 1, 2, 4, 1, 1, 2, 0, 0]
number_of_clustered_urllc_users_11_embb_users = [1, 2, 0, 1, 1, 1, 0, 0, 1, 1, 0]
number_of_failed_urllc_transmissions_11_embb_users = sum(number_of_clustered_urllc_users_11_embb_users) - sum(number_of_puncturing_users_11_embb_users)

# Separate bar plots for each metric
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
# Data Rate Plot
axes[0].bar(embb_users_11_users, users_data_rates_11_embb_users, color='blue')
axes[0].set_ylabel('Data Rate (bps)')
axes[0].set_xlabel('eMBB User Index')
axes[0].set_title('Data Rate per eMBB User')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Puncturing Users Plot
axes[1].bar(embb_users_11_users, number_of_puncturing_users_11_embb_users, color='orange')
axes[1].set_ylabel('Number of Transmissions')
axes[1].set_title('Number of Transmitting URLLC users per eMBB User')
axes[1].set_xlabel('eMBB User Index')
#axes[1].set_title('Puncturing Users per eMBB User')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Allocated RBs Plot
axes[2].bar(embb_users_11_users, number_of_allocated_RBs_11_embb_users, color='green')
axes[2].set_ylabel('Number of Allocated RBs')
axes[2].set_title('Number of Allocated RBs per eMBB User')
axes[2].set_xlabel('eMBB User Index')
#axes[2].set_title('Allocated RBs per eMBB User')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

# Clustered URLLC Users Plot
axes[3].bar(embb_users_11_users, number_of_clustered_urllc_users_11_embb_users, color='red')
axes[3].set_ylabel('Number of Clustered URLLC Users')
axes[3].set_xlabel('eMBB User Index')
axes[3].set_title('Number of Clustered URLLC Users per eMBB User')
#axes[1,1].set_xlabel('eMBB Users')
axes[3].grid(axis='y', linestyle='--', alpha=0.7)


# # Rotate x-axis labels
# for ax in axes:
#     ax.set_xticklabels(embb_users_11_users, rotation=45, ha='right')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 1)
#axes = axes.flatten()

number_of_embb_users = [3,7,11]
number_of_failed_urlllc_transmissions = [number_of_failed_urllc_transmissions_3_embb_users,number_of_failed_urllc_transmissions_7_embb_users,number_of_failed_urllc_transmissions_11_embb_users]
axes.plot(number_of_embb_users, number_of_failed_urlllc_transmissions, color='green', marker='o')
axes.set_ylabel('Number of failed URLLC transmissions')
axes.set_xlabel('Number of eMBB users in the network')
#axes[0].set_title('')
axes.grid()
plt.tight_layout()
plt.show()