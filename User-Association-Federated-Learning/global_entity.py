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
    def __init__(self, num_clients):
        #User_Equipment.__init__(self)
        self.global_entity_id = 1
        self.global_memory = DNN_TRAINING_MEMORY()
        self.local_models = []
        self.num_clients = num_clients
        self.rounds = 0

    def initialize_global_model(self,input_features_dim, output_features_dim):
        self.global_model = DNN(input_features_dim,output_features_dim)

    def perform_random_association(self,users):
        self.initial_associations = []
        for user in users:
            if len(user.access_points_within_radius) > 0:
                user_num_access_points = len(user.access_points_within_radius) 
                randint = int(random.randint(0,user_num_access_points-1))
                user_access_point = user.access_points_within_radius[randint]
                self.initial_associations.append((user.user_label, user_access_point))
            else: 
                user.access_points_within_radius.append((1,user.distances_from_access_point[0]))
                user_access_point = user.access_points_within_radius[0]
                self.initial_associations.append((user.user_label, user_access_point))

        return self.initial_associations
    
    def initialize_global_memory(self, max_samples, num_users, num_input_features, num_access_points):
        input_features = []
        user_associations = []
        sample_rewards = []
        
        #Random channel gains
        for x in range(0,max_samples):

            for y in range(0,num_users):

                user_id = y+1
                input_features.append(user_id)

                user_distance = random.random()
                input_features.append(user_distance)

                user_channel_gain = random.random()
                input_features.append(user_channel_gain)

            sample_rewards.append(random.randint(0,5))

            

        for x in range(0,max_samples):
            for y in range(0,num_users):
                user_association = random.random()
                user_associations.append(user_association)

        input_features = np.array(input_features).reshape(max_samples,num_input_features*num_users)
        user_associations = np.array(user_associations).reshape(max_samples,num_users)
        sample_rewards = np.array(sample_rewards)
        # print(user_associations)

        for x in range(0,max_samples):
            self.global_memory.add((input_features[x], user_associations[x], sample_rewards[x]))

        return self.global_memory
    
    def acquire_local_model(self, local_model):
        self.local_models.append(local_model)

    def clear_local_models_memory(self):
        if len(self.local_models) > 0:
            self.local_models.clear()

    def aggregate_local_models(self):
        self.rounds+=1
        # Federated averaging
        with torch.no_grad():
            new_state_dict = {}
            for key in self.global_model.state_dict():
                key_sum = sum(self.local_models[i].state_dict()[key] for i in range(self.num_clients))
                new_state_dict[key] = key_sum / self.num_clients
            self.global_model.load_state_dict(new_state_dict)

        

        