import random
import numpy as np
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
import numpy as np
from matplotlib.patches import Rectangle
import math
from State_Space import State_Space
from numpy import interp
import pandas as pd
from Communication_Channel import Communication_Channel
from DNN import DNN
from DNN_training_memory import DNN_TRAINING_MEMORY
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class GLOBAL_ENTITY():
    def __init__(self, num_access_point):
        #User_Equipment.__init__(self)
        self.global_entity_id = 1
        self.global_memory = []#DNN_TRAINING_MEMORY()
        self.local_models = []
        self.num_access_point = num_access_point
        self.rounds = 0
        self.local_associations = []
        self.global_reward = 0
        self.aggregate_count = 0
        self.local_associations_reset_count = 0
        self.global_reward_reset_count = 0

    def initialize_global_model(self,input_features_dim, output_features_dim):
        self.global_model = DNN(input_features_dim,output_features_dim)

    def perform_random_association(self,users):
        self.initial_associations = []
        for user in users:
            if len(user.access_points_within_radius) > 0:
                user_num_access_points = len(user.access_points_within_radius) 
                randint = int(random.randint(0,user_num_access_points-1))
                #randint = int(random.randint(0,2))
                user_access_point = user.access_points_within_radius[randint]
                self.initial_associations.append((user.user_label, user_access_point))
            else:
                index_of_min = np.argmin(user.distances_from_access_point) 
                user.access_points_within_radius.append((index_of_min+1,user.distances_from_access_point[index_of_min]))
                user_access_point = user.access_points_within_radius[0]
                self.initial_associations.append((user.user_label, user_access_point))

        return self.initial_associations
    
    def initialize_global_memory(self, max_samples, num_users, num_input_features, num_access_points):
        input_features = []
        user_associations = []
        sample_rewards = []

        num_samples = max_samples*num_access_points

        for x in range(0,num_access_points):
            self.global_memory.append(DNN_TRAINING_MEMORY())
        
        #Random channel gains
        for x in range(0,num_samples):

            for y in range(0,num_users):

                user_id = y+1
                input_features.append(user_id)

                user_channel_rate = random.random()
                input_features.append(user_channel_rate)

                # user_distance = random.random()
                # input_features.append(user_distance)

                # user_channel_gain = random.random()
                # input_features.append(user_channel_gain)

            sample_rewards.append(random.randint(0,5))

            

        for x in range(0,num_samples):
            for y in range(0,num_users):
                user_association = random.random()
                user_associations.append(user_association)

        input_features = np.array(input_features).reshape(num_samples,num_input_features*num_users)
        user_associations = np.array(user_associations).reshape(num_samples,num_users)
        sample_rewards = np.array(sample_rewards)

        #print('input_features: ', len(sample_rewards))

        batch_size = max_samples
        for i, memory_obj in enumerate(self.global_memory):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, num_samples)
            batch_data = [(input_features[j], user_associations[j], sample_rewards[j]) for j in range(start_index, end_index)]
            memory_obj.storage.extend(batch_data)

        #print('len global memory: ', len(self.global_memory[0].storage))

        return self.global_memory
    
    def acquire_local_model(self, local_model):
        self.local_models.append(local_model)

    def clear_local_models_memory(self):
        if len(self.local_models) > 0:
            self.local_models.clear()

    def aggregate_local_models(self):
        self.rounds+=1
        # Federated averaging
        global_model_state = self.local_models[0].state_dict()
        for model in self.local_models[1:]:
            model_state = model.state_dict()
            for key in global_model_state.keys():
                global_model_state[key] += model_state[key]  # Accumulate the weights
        # Calculate the average
        for key in global_model_state.keys():
            global_model_state[key] /= len(self.local_models)
        self.global_model.load_state_dict(global_model_state)
        self.aggregate_count+=1

    def acquire_local_user_associations(self, associations):
        self.local_associations.append(associations)

    def aggregate_user_associations(self):
        local_associations = np.array(self.local_associations)
        local_associations = np.sum(local_associations,axis=0)
        #self.local_associations = local_associations
        #local_associations = np.array([2,2,2,2,2,2,2,2,2,2,2,2])
        #print('local_associations aggregated: ', local_associations)
        return local_associations

    def clear_local_user_associations(self):
        self.local_associations_reset_count+=1
        #print(self.local_associations_reset_count)
        if self.local_associations_reset_count >= self.num_access_point:
            self.local_associations.clear()
            self.local_associations_reset_count = 0
            

    def calculate_global_reward(self, local_reward):
        self.global_reward+=local_reward

    def reset_global_reward(self):
        self.global_reward_reset_count+=1
        if self.global_reward_reset_count >= self.num_access_point:
            self.global_reward = 0
            self.global_reward_reset_count = 0



        

        
