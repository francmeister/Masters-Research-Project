import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
individual_large_scale_gains = np.load('individual_large_scale_gains.npy')
offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
RBs_actions = np.load('subcarrier_actions.npy')
individual_channel_rates = np.load('individual_channel_rate_rewards.npy')

# Check Data Shapes
num_users = individual_large_scale_gains.shape[1]

# 1. Correlate Gains with Actions (Scatter Plot)
plt.figure(figsize=(12, 6))
for user in range(num_users):
    plt.scatter(
        individual_large_scale_gains[:, user],
        offload_actions[:, user],
        alpha=0.7, label=f'User {user + 1}'
    )
plt.title('Correlation Between Large-Scale Gains and Offload Actions')
plt.xlabel('Large-Scale Gains')
plt.ylabel('Offload Actions')
plt.legend()
plt.show()

# 2. Action Trends with Rewards (Overlay Plot)
plt.figure(figsize=(12, 6))
for user in range(num_users):
    plt.plot(
        offload_actions[:, user],
        label=f'Offload Actions User {user + 1}', alpha=0.7
    )
    plt.plot(
        individual_channel_rates[:, user],
        linestyle='--', label=f'Channel Rates User {user + 1}', alpha=0.7
    )
plt.title('Offload Actions and Channel Rates Trends per User')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.show()

# Normalize Data (Min-Max Scaling)
def min_max_scale(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Apply normalization
normalized_large_scale_gains = min_max_scale(individual_large_scale_gains)
normalized_offload_actions = min_max_scale(offload_actions)
normalized_channel_rates = min_max_scale(individual_channel_rates)

# Compute mean values for each user
mean_large_scale_gains = np.mean(normalized_large_scale_gains, axis=0)
mean_offload_actions = np.mean(normalized_offload_actions, axis=0)
mean_channel_rates = np.mean(normalized_channel_rates, axis=0)

# Prepare for Bar Plot
users = [f'User {i + 1}' for i in range(num_users)]
bar_width = 0.2
x = np.arange(num_users)

# Plot Aggregated Bar Chart
plt.figure(figsize=(12, 6))
plt.bar(x, mean_large_scale_gains, width=bar_width, label='Normalized Large-Scale Gains')
plt.bar(x + bar_width, mean_offload_actions, width=bar_width, label='Normalized Offload Actions')
plt.bar(x + 2 * bar_width, mean_channel_rates, width=bar_width, label='Normalized Channel Rates')
plt.title('Normalized Mean Values Across Users')
plt.xlabel('Users')
plt.ylabel('Normalized Mean Value')
plt.xticks(x + bar_width, users)
plt.legend()
plt.show()


# 4. Correlation Heatmap for Gains and Actions Across Users
combined_data = np.hstack([
    individual_large_scale_gains, offload_actions, power_actions
])
correlation_matrix = np.corrcoef(combined_data.T)  # Transpose for correlation
sns.heatmap(
    correlation_matrix, annot=False, cmap='coolwarm',
    xticklabels=(["L-Gain U" + str(i + 1) for i in range(num_users)] +
                 ["O-Action U" + str(i + 1) for i in range(num_users)] +
                 ["P-Action U" + str(i + 1) for i in range(num_users)]),
    yticklabels=(["L-Gain U" + str(i + 1) for i in range(num_users)] +
                 ["O-Action U" + str(i + 1) for i in range(num_users)] +
                 ["P-Action U" + str(i + 1) for i in range(num_users)])
)
plt.title('Correlation Matrix of Gains, Offload, and Power Actions')
plt.show()
