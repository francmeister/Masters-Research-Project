import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')
users_achieved_channel_rates_AP_1 = np.load('users_achieved_channel_rates_AP_1.npy')
users_achieved_channel_rates_AP_2 = np.load('users_achieved_channel_rates_AP_2.npy')
users_achieved_channel_rates_AP_3 = np.load('users_achieved_channel_rates_AP_3.npy')

users_distances_to_associated_APs_AP_1 = np.load('users_distances_to_associated_APs_AP_1.npy')
users_distances_to_associated_APs_AP_2 = np.load('users_distances_to_associated_APs_AP_2.npy')
users_distances_to_associated_APs_AP_3 = np.load('users_distances_to_associated_APs_AP_3.npy')

users_distances_to_other_APs_AP_1 = np.load('users_distances_to_other_APs_AP_1.npy')
users_distances_to_other_APs_AP_2 = np.load('users_distances_to_other_APs_AP_2.npy')
users_distances_to_other_APs_AP_3 = np.load('users_distances_to_other_APs_AP_3.npy')

users_channel_rates_to_other_APs_AP_1 = np.load('users_channel_rates_to_other_APs_AP_1.npy')
users_channel_rates_to_other_APs_AP_2 = np.load('users_channel_rates_to_other_APs_AP_2.npy')
users_channel_rates_to_other_APs_AP_3 = np.load('users_channel_rates_to_other_APs_AP_3.npy')

access_point_users_AP_1 = np.load('access_point_users_AP_1.npy')
access_point_users_AP_2 = np.load('access_point_users_AP_2.npy')
access_point_users_AP_3 = np.load('access_point_users_AP_3.npy')

# print('access_point_users_AP_1')
# print(access_point_users_AP_1)
# print('----------------------------------')
# print('users_achieved_channel_rates_AP_1')
# print(users_achieved_channel_rates_AP_1)
# print('----------------------------------')
# print('users_distances_to_associated_APs_AP_1')
# print(users_distances_to_associated_APs_AP_1)
# print('----------------------------------')
# print('users_distances_to_other_APs_AP_1')
# print(users_distances_to_other_APs_AP_1)
# print('----------------------------------')
# print('users_channel_rates_to_other_APs_AP_1')
# print(users_channel_rates_to_other_APs_AP_1)

print('access_point_users_AP_2')
print(access_point_users_AP_2)
# print('----------------------------------')
# print('users_achieved_channel_rates_AP_2')
# print(users_achieved_channel_rates_AP_2)
# print('----------------------------------')
# print('users_distances_to_associated_APs_AP_2')
# print(users_distances_to_associated_APs_AP_2)
# print('----------------------------------')
# print('users_distances_to_other_APs_AP_2')
# print(users_distances_to_other_APs_AP_2)
# print('----------------------------------')
# print('users_channel_rates_to_other_APs_AP_2')
# print(users_channel_rates_to_other_APs_AP_2)

if len(access_point_users_AP_1) > 0:
    access_point_1_users = access_point_users_AP_1[:,1]
else:
    access_point_1_users = []

if len(access_point_users_AP_2) > 0:
    access_point_2_users = access_point_users_AP_2[:,1]
else:
    access_point_2_users = []

if len(access_point_users_AP_3) > 0:
    access_point_3_users = access_point_users_AP_3[:,1]
else:
    access_point_3_users = []

print('access_point_3_users: ', access_point_3_users)
#print(users_distances_to_other_APs_AP_1[0])#users_distances_to_other_APs_AP_1[0][0][0]

access_points = users_distances_to_other_APs_AP_1[0][:,1]

channel_rates_to_access_points = []#[users_distances_to_other_APs_AP_1[0][:,2]


def plot_per_user(user_id, access_point_1_users, access_point_2_users, access_point_3_users):

    user_associated_AP = 0
    if user_id in access_point_1_users:
        user_associated_AP = 1
        index = np.where(access_point_1_users == user_id)[0][0]
        channel_rates_to_access_points = users_channel_rates_to_other_APs_AP_1[index][:,2]
    elif user_id in access_point_2_users:
        user_associated_AP = 2
        index = np.where(access_point_2_users == user_id)[0][0]
        channel_rates_to_access_points = users_channel_rates_to_other_APs_AP_2[index][:,2]
    elif user_id in access_point_3_users:
        user_associated_AP = 3
        index = np.where(access_point_3_users == user_id)[0][0]
        channel_rates_to_access_points = users_channel_rates_to_other_APs_AP_3[index][:,2]
    print('user_associated_AP: ', user_associated_AP)


    default_color = 'skyblue'
    association_color = 'red'
    colors = []
    labels = []
    for access_point in access_points:
        if access_point == user_associated_AP:
            colors.append(association_color)
            labels.append('Associated AP')
        else:
            colors.append(default_color)
            labels.append('Other non-associated APs')

    bars = plt.bar(access_points, channel_rates_to_access_points, color=colors)

    # Create custom legend
    associated_bar = plt.Rectangle((0, 0), 1, 1, fc=association_color)
    non_associated_bar = plt.Rectangle((0, 0), 1, 1, fc=default_color)
    plt.legend([associated_bar, non_associated_bar], ['Associated AP', 'Other non-associated APs'])


    # Adding labels and title
    plt.xlabel('Access Point Number')
    plt.ylabel('Channel Rate to AP')
    plt.title('User: ' + str(user_id))
    plt.show()

user_id = 12
plot_per_user(user_id,access_point_1_users,access_point_2_users,access_point_3_users)