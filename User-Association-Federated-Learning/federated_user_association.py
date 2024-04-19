import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import gym
import time
import threading
import multiprocessing
from eMBB_UE import eMBB_UE
from URLLC_UE import URLLC_UE
from SBS import SBS
from Communication_Channel import Communication_Channel
from numpy import interp
import math
from NetworkEnv import NetworkEnv
from global_entity import GLOBAL_ENTITY

#Create a Square grid of overlapping Access Points (3 APs) - 0 to 100 km both x and y

x_grid = 100
y_grid = 100
access_point_radius = 70

num_embb_users = 6
num_urllc_users = 6
all_users = []
user_count = 1
embb_user_count = 1
urllc_user_count = 1
access_point_count = 1
access_points = []
num_access_points = 3
num_users = num_embb_users+num_urllc_users
num_input_features_per_user = 3
num_input_features = num_users*num_input_features_per_user
num_output_features = num_users
max_samples = 20
comm_channel = Communication_Channel(1)

for x in range(0,num_embb_users):
   embb_user = eMBB_UE(embb_user_count,user_count,100,600)
   all_users.append(embb_user)
   embb_user_count+=1
   user_count+=1

for x in range(0,num_urllc_users):
   urllc_user = URLLC_UE(urllc_user_count,user_count,100,600)
   all_users.append(urllc_user)
   urllc_user_count+=1
   user_count+=1

for x in range(0,num_access_points):
   access_point = SBS(access_point_count,num_access_points,num_input_features,num_output_features)
   access_points.append(access_point)
   access_point_count+=1

access_point_coordinates = []
for access_point in range(0, num_access_points):
   x_coord = random.randint(0, x_grid)
   y_coord = random.randint(0, y_grid)
   access_point_coordinates.append((x_coord,y_coord))

user_coordinates = []
for user in range(0, num_users):
   x_coord = random.randint(0, x_grid)
   y_coord = random.randint(0, y_grid)
   user_coordinates.append((x_coord,y_coord))

coord_index = 0
for access_point in access_points:
   access_point.set_coordinates(access_point_coordinates[coord_index])
   coord_index+=1

coord_index = 0
for user in all_users:
   user.set_coordinates(user_coordinates[coord_index])
   user.calculate_distances_from_access_point(access_point_coordinates,access_point_radius)
   coord_index+=1


for access_point in access_points:
   access_point.find_users_within_distance_radius(access_point_radius, all_users)

# for user in all_users:
#    print('user.distances_from_access_point')
#    print(user.distances_from_access_point)

# for user in all_users:
#    print('user.access_points_within_radius')
#    print(user.access_points_within_radius)


global_entity = GLOBAL_ENTITY(num_users)
global_entity.initialize_global_model(num_input_features,num_output_features)
global_memory = global_entity.initialize_global_memory(max_samples,num_users,num_input_features_per_user,num_access_points)
initial_user_associations = global_entity.perform_random_association(all_users)
#print(global_memory.storage[0])

for access_point in access_points:
   access_point.get_all_users(all_users)
   access_point.initialize_DNN_model(global_entity.global_model)
   access_point.acquire_global_memory(global_entity.global_memory)


for access_point in access_points:
   access_point_users = []
   for user in all_users:
      for association in initial_user_associations:
         #print(association)
         if association[0] == user.user_label and association[1][0] == access_point.SBS_label:
            user.distance_from_associated_access_point = association[1][1]
            access_point_users.append(user)

   access_point.associate_users(access_point_users)

channel_gains = []
for x in range(0,3):
   for access_point in access_points:
      access_point.train_local_dnn()

   for access_point in access_points:
      global_entity.acquire_local_model(access_point.access_point_model)

   global_entity.aggregate_local_models()

   for access_point in access_points:
      access_point.acquire_global_model(global_entity.global_model)

   for access_point in access_points:
      global_entity.acquire_local_user_associations(access_point.predict_future_association(access_point_radius))
      global_entity.calculate_global_reward(20)

   user_association = global_entity.aggregate_user_associations()

   for access_point in access_points:
      access_point.reassociate_users(user_association)
      access_point.populate_buffer_memory_sample_with_reward(global_entity.global_reward)


   global_entity.clear_local_models_memory()
   global_entity.clear_local_user_associations()
   global_entity.reset_global_reward()






#channel_gains = access_points[0].predict_future_association(access_point_radius)


# print('access_points[0].users')
# for user in access_points[0].users:
#    print(user.user_label)

# print('')
# for user in access_points[1].users:
#    print(user.user_label)

# print('')
# for user in access_points[2].users:
#    print(user.user_label)

# print('global memory')
# print(global_memory.storage)

for access_point in access_points:
   access_point.acquire_global_model(global_entity.global_model)
   access_point.acquire_global_memory(global_memory)











