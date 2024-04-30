#from DNN import DNN
from DNN_training_memory import DNN_TRAINING_MEMORY
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from numpy import interp
import math
import matplotlib.pyplot as plt

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
         super(DNN, self).__init__()
         self.fc1 = nn.Linear(input_dim, 100)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(100, 100)
         self.fc3 = nn.Linear(100, output_dim)
         #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

user_id = 1

distance_to_AP_1 = 0.000000005
distance_to_AP_2 = 0.001
distance_to_AP_3 = 2000

num_input_features = 3
num_access_points = 3
num_users = 1

dnn = DNN(num_access_points,num_users)
training_memory = DNN_TRAINING_MEMORY()

# Generate random samples for initial training
user_ids = []
distances = []
channel_gains = []
input_features = []
rewards = []
user_associations = []
buffer_memory = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for x in range(0,40):
    user_id = 1
    distance = random.random()
    channel_gain = random.random()
    reward = random.random()
    user_association = random.randint(1,3)

    user_ids.append(user_id)
    distances.append(distance)
    channel_gains.append(channel_gain)
    rewards.append(reward)
    user_associations.append(user_association)

for x in range(0,40):
    input_features.append([user_ids[x], distances[x], channel_gains[x]])

input_features = np.array(input_features)
user_associations = np.array(user_associations)
rewards = np.array(rewards)

#print(rewards)

for x in range(0,40):
    training_memory.add((input_features[x], user_associations[x], rewards[x]))

def preprocessing(distance, channel_gain):
    features = []
    distance_normalized = interp(distance,[0,2000],[0,500])
    channel_gain_normalized = interp(channel_gain,[0,10],[0,1])

    features.append(1)
    features.append(distance_normalized)
    features.append(channel_gain_normalized)

    features = np.array(features)
    return features

def calculate_channel_rate(distance, channel_gain):
    channel_rate_numerator = 400*math.pow(distance,-1)*channel_gain
    channel_rate_denominator = 1
    RB_bandwidth = 12000
    channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))

    return channel_rate/1000

def prediction_future_association(distance, channel_gain):
    preprocessed_inputs = preprocessing(distance, channel_gain)
    preprocessed_inputs_tensor = torch.Tensor(preprocessed_inputs).to(device)
    association_prediction = dnn(preprocessed_inputs_tensor)
    association_prediction = association_prediction.detach().numpy()
    #print(association_prediction[0])
    association_prediction = round(association_prediction[0])

    if association_prediction > 3:
        association_prediction = 3
    elif association_prediction < 1:
        association_prediction = 1

    buffer_memory.append((preprocessed_inputs,association_prediction,0))

    return association_prediction

def populate_buffer_memory_sample_with_reward(current_association_reward):
    rewards_in_memory = []
    if len(buffer_memory) > 1:
        new_sample = (buffer_memory[0][0],buffer_memory[0][1],current_association_reward)
        buffer_memory[0] = new_sample
        dnn_memory_rewards = []
        for sample in training_memory.storage:
            dnn_memory_rewards.append(sample[2])
        max_index = dnn_memory_rewards.index(max(dnn_memory_rewards))

        if current_association_reward >= dnn_memory_rewards[max_index]:
            training_memory.add(buffer_memory[0])
            #print('SBS: ', self.SBS_label, 'Appended')

        buffer_memory.pop(0)
    for sample in training_memory.storage:
        rewards_in_memory.append(sample[2])
    
    average_reward_in_memory = sum(rewards_in_memory)/len(rewards_in_memory) + random.random()
    return average_reward_in_memory



current_association = 3
current_distance = distance_to_AP_3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dnn.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(dnn.parameters(), lr=0.001)
num_training_epochs = 100
training_loss = []
average_rewards_in_memory = []
channel_rates = []
training_loss = []

for x in range(0,3000):

    # x_train, y_train, sample_rewards = training_memory.sample(20)
    # y_train = y_train.reshape(20,1)
    # x_train_tensor = torch.Tensor(x_train).to(device)
    # y_train_tensor = torch.Tensor(y_train).to(device)

    # if x_train_tensor.dtype != dnn.fc1.weight.dtype:
    #     x_train_tensor = x_train_tensor.to(dnn.fc1.weight.dtype)
    #     y_train_tensor = y_train_tensor.to(dnn.fc1.weight.dtype)

    # for epoch in range(num_training_epochs):
    #         y_pred_tensor = dnn(x_train_tensor)
    #         loss = criterion(y_pred_tensor, y_train_tensor)
    #         training_loss.append(loss.detach().numpy())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    # if current_association == 1:
    #     current_distance = distance_to_AP_1
    # elif current_association == 2:
    #     current_distance = distance_to_AP_2
    # elif current_association == 3:
    #     current_distance = distance_to_AP_3


    current_channel_gain = np.random.exponential(1)

    if x < 1000:
        current_distance = distance_to_AP_1
    elif x >= 1000 and x < 2000: 
        current_distance = distance_to_AP_2
    elif x >=2000:
        current_distance = distance_to_AP_3
    #print(current_distance)
    current_channel_rate = calculate_channel_rate(current_distance,current_channel_gain)
    channel_rates.append(current_channel_rate)
    # #print(current_channel_rate)

    # future_association = prediction_future_association(current_distance, current_channel_gain)
    # #print(future_association)
    # current_association = future_association

    # average_reward_in_memory = populate_buffer_memory_sample_with_reward(current_channel_rate)
    # average_rewards_in_memory.append(average_reward_in_memory)

#print(average_rewards_in_memory)


timesteps_average_rewards_in_memory = []
timesteps_channel_rates = []
timesteps_training_loss = []

x = 0
for gb in average_rewards_in_memory:
    timesteps_average_rewards_in_memory.append(x)
    x+=1

x = 0
for gb in channel_rates:
    timesteps_channel_rates.append(x)
    x+=1

x = 0
for gb in training_loss:
    timesteps_training_loss.append(x)
    x+=1


#print(average_rewards_in_memory)
#plt.plot(timesteps_average_rewards_in_memory, average_rewards_in_memory, color ="blue")
plt.plot(timesteps_channel_rates, channel_rates, color ="blue")
#plt.plot(timesteps_training_loss, training_loss, color ="blue")

plt.show()





