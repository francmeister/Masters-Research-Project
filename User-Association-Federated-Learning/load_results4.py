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

users_access_points_within_radius_AP_1 = np.load('users_access_points_within_radius_AP_1.npy')
users_access_points_within_radius_AP_2 = np.load('users_access_points_within_radius_AP_2.npy')
users_access_points_within_radius_AP_3 = np.load('users_access_points_within_radius_AP_3.npy')

print('users_access_points_within_radius_AP_1')
print(users_access_points_within_radius_AP_1)
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

# print('access_point_users_AP_2')
# print(access_point_users_AP_2)
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

distances_to_access_points = []#[users_distances_to_other_APs_AP_1[0][:,2]
access_points_in_radius = []

def plot_per_user(user_id, access_point_1_users, access_point_2_users, access_point_3_users):

    user_associated_AP = 0
    if user_id in access_point_1_users:
        user_associated_AP = 1
        index = np.where(access_point_1_users == user_id)[0][0]
        distances_to_access_points = users_distances_to_other_APs_AP_1[index][:,2]
        access_points_in_radius = users_access_points_within_radius_AP_1[index][:,2]
    elif user_id in access_point_2_users:
        user_associated_AP = 2
        index = np.where(access_point_2_users == user_id)[0][0]
        distances_to_access_points = users_distances_to_other_APs_AP_2[index][:,2]
        access_points_in_radius = users_access_points_within_radius_AP_2[index][:,2]
    elif user_id in access_point_3_users:
        user_associated_AP = 3
        index = np.where(access_point_3_users == user_id)[0][0]
        distances_to_access_points = users_distances_to_other_APs_AP_3[index][:,2]
        access_points_in_radius = users_access_points_within_radius_AP_3[index][:,2]
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

    bars = plt.bar(access_points, distances_to_access_points, color=colors)

    for bar, label in zip(bars, access_points_in_radius):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 
                 str(label), ha='center', va='bottom', fontsize=10, color='black')

    # Create custom legend
    associated_bar = plt.Rectangle((0, 0), 1, 1, fc=association_color)
    non_associated_bar = plt.Rectangle((0, 0), 1, 1, fc=default_color)
    plt.legend([associated_bar, non_associated_bar], ['Associated AP', 'Other non-associated APs'])

    # plt.text(1.05, 0.9, '1 = Within 100m radius\n0 = Outside 100m radius',
    #          transform=plt.gca().transAxes, fontsize=10, color='black', ha='left')


    # Adding labels and title
    plt.xlabel('Access Point Number')
    plt.ylabel('Distances to Access Points (m)')
    plt.title('User: ' + str(user_id))
    plt.show()


user_id =1
plot_per_user(user_id,access_point_1_users,access_point_2_users,access_point_3_users)