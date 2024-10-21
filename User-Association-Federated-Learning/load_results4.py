import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
users_achieved_channel_rates_AP_1 = np.load('users_achieved_channel_rates_AP_1.npy')
# users_achieved_channel_rates_AP_2 = np.load('users_achieved_channel_rates_AP_2.npy')
# users_achieved_channel_rates_AP_3 = np.load('users_achieved_channel_rates_AP_3.npy')

users_distances_to_associated_APs_AP_1 = np.load('users_distances_to_associated_APs_AP_1.npy')
# users_distances_to_associated_APs_AP_2 = np.load('users_distances_to_associated_APs_AP_2.npy')
# users_distances_to_associated_APs_AP_3 = np.load('users_distances_to_associated_APs_AP_3.npy')

users_distances_to_other_APs_AP_1 = np.load('users_distances_to_other_APs_AP_1.npy')
# users_distances_to_other_APs_AP_2 = np.load('users_distances_to_other_APs_AP_2.npy')
# users_distances_to_other_APs_AP_3 = np.load('users_distances_to_other_APs_AP_3.npy')

users_channel_rates_to_other_APs_AP_1 = np.load('users_channel_rates_to_other_APs_AP_1.npy')
# users_channel_rates_to_other_APs_AP_2 = np.load('users_channel_rates_to_other_APs_AP_2.npy')
# users_channel_rates_to_other_APs_AP_3 = np.load('users_channel_rates_to_other_APs_AP_3.npy')


print('users_achieved_channel_rates_AP_1')
print(users_achieved_channel_rates_AP_1)
print('----------------------------------')
print('users_distances_to_associated_APs_AP_1')
print(users_distances_to_associated_APs_AP_1)
print('----------------------------------')
print('users_distances_to_other_APs_AP_1')
print(users_distances_to_other_APs_AP_1)
print('----------------------------------')
print('users_channel_rates_to_other_APs_AP_1')
print(users_channel_rates_to_other_APs_AP_1)
