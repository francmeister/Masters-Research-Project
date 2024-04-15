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

class GLOBAL_ENTITY():
    def __init__(self):
        #User_Equipment.__init__(self)
        self.global_entity_id = 1
        self.global_memory = DNN_TRAINING_MEMORY()
        
    def initialize_global_model(self,input_features_dim, output_features_dim):
        self.global_model = DNN(input_features_dim,output_features_dim)

    def perform_random_association(self,users):
        self.initial_associations = []
        for user in users:
            user_num_access_points = len(user.access_points_within_radius) 
            randint = int(random.randint(0,user_num_access_points-1))
            user_access_point = user.access_points_within_radius[randint]
            self.initial_associations.append((user.user_label, user_access_point))
        return self.initial_associations
    
    def initialize_global_memory(self, max_samples, num_users, num_input_features, num_access_points):
        input_features = []
        user_associations = []
        
        #Random channel gains
        for x in range(0,max_samples):

            for y in range(0,num_users):

                user_id = y+1
                input_features.append(user_id)

                user_distance = random.random()
                input_features.append(user_distance)

                user_channel_gain = random.random()
                input_features.append(user_channel_gain)

                user_queue_state = random.random()
                input_features.append(user_queue_state)

            input_features.append(random.randint(0,5))

            

        for x in range(0,max_samples):
            for y in range(0,num_users):
                user_association = random.random()
                user_associations.append(user_association)

        input_features = np.array(input_features).reshape(max_samples,num_input_features*num_users+1)
        user_associations = np.array(user_associations).reshape(max_samples,num_users)

        for x in range(0,max_samples):
            self.global_memory.add((input_features[x], user_associations[x]))

        return self.global_memory

        

        
