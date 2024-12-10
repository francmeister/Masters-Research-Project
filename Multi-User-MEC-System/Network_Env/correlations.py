# import numpy as np
# import matplotlib.pyplot as plt


# individual_battery_energy_levels = np.load('individual_battery_energy_levels.npy')
# individual_large_scale_gains = np.load('individual_large_scale_gains.npy')
# individual_small_scale_gains = np.load('individual_small_scale_gains.npy')
# individual_local_queue_lengths = np.load('individual_local_queue_lengths.npy')
# individual_offload_queue_lengths = np.load('individual_offload_queue_lengths.npy')
# individual_channel_rates = np.load('individual_channel_rate_rewards.npy')

# offload_actions = np.load('offloading_actions.npy')
# power_actions = np.load('power_actions.npy')
# RBs_actions = np.load('subcarrier_actions.npy')

# print('individual_large_scale_gains')
# print(individual_large_scale_gains)

# individual_large_scale_gains
# [[3.86746433e-10 4.79082753e-06 6.57127535e-10]
#  [6.40523374e-10 5.00950853e-06 2.52563258e-10]
#  [5.89289593e-10 3.30324866e-06 5.12342272e-10]
#  ...
#  [6.00923956e-10 2.96829038e-06 1.43288436e-10]
#  [1.13371694e-09 3.56819111e-06 4.35865589e-10]
#  [8.00151996e-10 1.09674413e-06 8.18135817e-11]]


# import numpy as np

# def compute_user_correlations(large_scale_gains, channel_rates):
#     """
#     Compute correlations between channel rates and large-scale channel gains for each user.

#     Parameters:
#         large_scale_gains (np.ndarray): 2D array where each column represents large-scale gains for a user.
#         channel_rates (np.ndarray): 2D array where each column represents channel rates for a user.

#     Returns:
#         list: A list of correlation coefficients for each user.
#     """
#     # Check if input matrices have the same dimensions
#     if large_scale_gains.shape != channel_rates.shape:
#         raise ValueError("large_scale_gains and channel_rates must have the same dimensions.")
    
#     num_users = large_scale_gains.shape[1]
#     correlations = []
    
#     for user in range(num_users):
#         gains = large_scale_gains[:, user]
#         rates = channel_rates[:, user]
        
#         # Compute correlation, handling cases where variance is zero
#         if np.std(gains) == 0 or np.std(rates) == 0:
#             correlation = 0  # Correlation is undefined if one array has zero variance
#         else:
#             correlation = np.corrcoef(gains, rates)[0, 1]
        
#         correlations.append(correlation)
    
#     return correlations


# #correlations = compute_user_correlations(individual_large_scale_gains,individual_channel_rates)
# #print(correlations)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
individual_large_scale_gains = np.load('individual_large_scale_gains.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
RBs_actions = np.load('subcarrier_actions.npy')

# Check Shapes
print("Shape of individual_large_scale_gains:", individual_large_scale_gains.shape)  # (Samples, Users)
print("Shape of offload_actions:", offload_actions.shape)  # (Samples, Users)
print("Shape of power_actions:", power_actions.shape)  # (Samples, Users)
print("Shape of RBs_actions:", RBs_actions.shape)  # (Samples, Users)

# Number of users (assuming the data is shaped as (Samples, Users))
num_users = individual_large_scale_gains.shape[1]

# Plot Large-Scale Gains for Each User
plt.figure(figsize=(12, 6))
for user in range(num_users):
    plt.plot(
        individual_large_scale_gains[:, user],
        label=f'User {user + 1}', alpha=0.7
    )
plt.title('Large-Scale Gains for Each User')
plt.xlabel('Sample Index')
plt.ylabel('Gain Value')
plt.legend()
plt.show()

# Plot Action Distributions for Each User
plt.figure(figsize=(12, 6))
for user in range(num_users):
    plt.hist(
        offload_actions[:, user],
        bins=20, alpha=0.5, label=f'Offload User {user + 1}'
    )
plt.title('Offload Actions Distribution per User')
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Compare Actions Across Users (Example for Power Actions)
plt.figure(figsize=(12, 6))
for user in range(num_users):
    plt.plot(
        power_actions[:, user],
        label=f'Power User {user + 1}', alpha=0.7
    )
plt.title('Power Actions for Each User')
plt.xlabel('Sample Index')
plt.ylabel('Power Action Value')
plt.legend()
plt.show()

# Correlation Heatmap Between Users
# (Flatten each user's data into a vector)
user_data = {
    f"User {user + 1}": individual_large_scale_gains[:, user] for user in range(num_users)
}
correlation_matrix = np.corrcoef([user_data[key] for key in user_data])
sns.heatmap(
    correlation_matrix, annot=True, xticklabels=user_data.keys(),
    yticklabels=user_data.keys(), cmap='coolwarm'
)
plt.title('Correlation of Large-Scale Gains Between Users')
plt.show()
