import numpy as np
import matplotlib.pyplot as plt


individual_battery_energy_levels = np.load('individual_battery_energy_levels.npy')
individual_large_scale_gains = np.load('individual_large_scale_gains.npy')
individual_small_scale_gains = np.load('individual_small_scale_gains.npy')
individual_local_queue_lengths = np.load('individual_local_queue_lengths.npy')
individual_offload_queue_lengths = np.load('individual_offload_queue_lengths.npy')
individual_channel_rates = np.load('individual_channel_rate_rewards.npy')

offload_actions = np.load('offloading_actions.npy')
power_actions = np.load('power_actions.npy')
RBs_actions = np.load('subcarrier_actions.npy')

# print('individual_large_scale_gains')
# print(individual_large_scale_gains)

# print('individual_channel_rates')
# print(individual_channel_rates)

import numpy as np

def compute_user_correlations(large_scale_gains, channel_rates):
    """
    Compute correlations between channel rates and large-scale channel gains for each user.

    Parameters:
        large_scale_gains (np.ndarray): 2D array where each column represents large-scale gains for a user.
        channel_rates (np.ndarray): 2D array where each column represents channel rates for a user.

    Returns:
        list: A list of correlation coefficients for each user.
    """
    # Check if input matrices have the same dimensions
    if large_scale_gains.shape != channel_rates.shape:
        raise ValueError("large_scale_gains and channel_rates must have the same dimensions.")
    
    num_users = large_scale_gains.shape[1]
    correlations = []
    
    for user in range(num_users):
        gains = large_scale_gains[:, user]
        rates = channel_rates[:, user]
        
        # Compute correlation, handling cases where variance is zero
        if np.std(gains) == 0 or np.std(rates) == 0:
            correlation = 0  # Correlation is undefined if one array has zero variance
        else:
            correlation = np.corrcoef(gains, rates)[0, 1]
        
        correlations.append(correlation)
    
    return correlations


correlations = compute_user_correlations(individual_large_scale_gains,individual_channel_rates)
print(correlations)