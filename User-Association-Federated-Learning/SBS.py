import pygame, sys, time, random
pygame.init()
import math
import numpy as np
from numpy import interp
from DNN import DNN
from DNN_training_memory import DNN_TRAINING_MEMORY
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import copy
import scipy.stats as stats

class SBS():
    def __init__(self, SBS_label, num_access_points, input_dim, output_dim):
        self.SBS_label = SBS_label
        self.num_access_points = num_access_points
        #SBS Telecom properties
        self.x_position = 200
        self.y_position = 200
        self.individual_rewards = []
        self.training_memory = DNN_TRAINING_MEMORY()
        self.access_point_model = DNN(input_dim,output_dim)
        self.buffer_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_update_tracker = 0
        self.tau = 0.1
        self.timestep_counter = 0
        self.average_reward_in_memory = 0
        self.training_loss = []
        self.set_properties()

    def get_all_users(self, all_users):
        self.all_users = copy.deepcopy(all_users)

    def associate_users(self, users):
        self.users = copy.deepcopy(users)
        # print('SBS_label')
        # print(self.SBS_label)
        # print('number of users')
        # print(self.users)
        self.embb_users = []
        self.urllc_users = []
        associated_users = []
        for user in self.users:
            if user.type_of_user_id == 0:
                self.embb_users.append(user)
                associated_users.append(user.user_label)
            elif user.type_of_user_id == 1:
                self.urllc_users.append(user)

        #print('SBS: ', self.SBS_label, 'associated users for next time slot: ', associated_users)


    def reassociate_users(self,user_association_matrix):
        #print(user_association_matrix)
        self.users.clear()
        self.embb_users.clear()
        self.urllc_users.clear()
        associated_users = []
        #print('SBS: ', self.SBS_label, 'user association matrix: ', user_association_matrix)
        for user in self.all_users:
            count = 0
            for user1 in user_association_matrix:
                if user.user_label == count+1:
                    if user1 == self.SBS_label:
                        self.users.append(user)

                count+=1


        for user in self.users:
            if user.type_of_user_id == 0:
                self.embb_users.append(user)
                associated_users.append(user.user_label)
            elif user.type_of_user_id == 1:
                self.urllc_users.append(user)   

        #print('SBS: ', self.SBS_label, 'associated users for next time slot: ', associated_users)

         
   

    def initialize_DNN_model(self,global_model):
        self.access_point_model.load_state_dict(global_model.state_dict()) 

    def get_SBS_center_pos(self):
        self.x_position = 200
        self.y_position = 200

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.x_coordinate = coordinates[0]
        self.y_coordinate = coordinates[1]

    def find_users_within_distance_radius(self,radius,users):
        SBS_label = self.SBS_label
        self.users_within_distance_radius = []
        for user in users:
            if user.distances_from_access_point[SBS_label-1] <= radius:
                self.users_within_distance_radius.append((user.user_label,user.distances_from_access_point[SBS_label-1]))

    def calculate_user_association_channel_rates(self,users,communication_channel):
        self.users_within_radius_channel_rates = []
        for user_within_distance_radius in self.users_within_distance_radius:
            for user in users:
                if user_within_distance_radius[0] == user.user_label:
                    channel_gain = user.calculate_user_association_channel_gains()
                    channel_rate = self.calculate_user_association_channel_rate(channel_gain,user_within_distance_radius[1], communication_channel,user.max_transmission_power_dBm)
                    self.users_within_radius_channel_rates.append((user_within_distance_radius[0],channel_rate))

    def calculate_user_association_channel_rate(self,channel_gain,distance,communication_channel,max_transmission_power):

        system_bandwidth_Hz = communication_channel.system_bandwidth_Hz
        noise_spectral_density = communication_channel.noise_spectral_density_W
        channel_rate_numerator = max_transmission_power*channel_gain*math.pow(distance,self.distance_exponent)
        channel_rate_denominator = noise_spectral_density#*RB_bandwidth
        channel_rate = (system_bandwidth_Hz*math.log2(1+(channel_rate_numerator/channel_rate_denominator)))

        return (channel_rate/1000)
    
    #def predict_future_association(self):

    def acquire_global_model(self, global_model):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_update_tracker == 0:
            self.access_point_model.load_state_dict(global_model.state_dict())
            self.model_update_tracker+=1
        else:
            # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
            for global_model_param, local_model_param in zip(global_model.parameters(), self.access_point_model.parameters()):
                local_model_param.data.copy_(self.tau * global_model_param.data + (1 - self.tau) * local_model_param.data)
            #self.access_point_model.to(device)

    def acquire_global_memory(self, global_memory):    
        self.training_memory = copy.deepcopy(global_memory[self.SBS_label-1])   

    def train_local_dnn(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.access_point_model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.access_point_model.parameters(), lr=0.001)
        self.num_training_epochs = 300
        x_train, y_train, sample_rewards = self.training_memory.sample(20)
        # print('len(x_train[0]): ', y_train[0])
        #print(len(x_train[0]))

        x_train_tensor = torch.Tensor(x_train).to(device)
        y_train_tensor = torch.Tensor(y_train).to(device)

        if x_train_tensor.dtype != self.access_point_model.fc1.weight.dtype:
            x_train_tensor = x_train_tensor.to(self.access_point_model.fc1.weight.dtype)
            y_train_tensor = y_train_tensor.to(self.access_point_model.fc1.weight.dtype)

        #print('Starting training of local DNN of Access Point: ', self.SBS_label)
        for epoch in range(self.num_training_epochs):
            # for i in range(0,len(x_train_tensor)):
            #     y_pred_tensor = self.access_point_model(x_train_tensor[i])
            #     loss = self.criterion(y_pred_tensor, y_train_tensor[i])
            #     print(loss)
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()

            y_pred_tensor = self.access_point_model(x_train_tensor)
            loss = self.criterion(y_pred_tensor, y_train_tensor)
            self.training_loss.append(loss.detach().numpy())
            #print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #print('Finished training local DNN of Access Point: ', self.SBS_label)
        # if self.SBS_label == 1:
        #     print(self.training_loss)
        return self.access_point_model

        # y_pred = self.access_point_model(x_train_tensor)
        # print('y_train_tensor[0]')
        # print(y_train_tensor[0])

        # print('y_pred[0]')
        # print(y_pred[0])

    def preprocess_model_inputs(self, access_point_radius):
        # input_features.append(user_id)

        # user_distance = random.random()
        # input_features.append(user_distance)

        # user_channel_gain = random.random()
        # input_features.append(user_channel_gain)

        user_ids = []
        user_distances = []
        user_channel_gains = []
        associated_users_ids = []
        for user in self.users:
            associated_users_ids.append(user.user_label)

        for user in self.all_users:
            user_ids.append(user.user_label)
            if user.user_label in associated_users_ids:
                user_distances.append(user.distance_from_associated_access_point)
                user_channel_gains.append(user.calculate_user_association_channel_gains())
            else:
                user_distances.append(0)
                user_channel_gains.append(0)

        user_distances_normalized = []
        for user_distance in user_distances:
            user_distances_normalized.append(interp(user_distance,[0,access_point_radius],[0,10]))

        user_channel_gains_normalized = []
        for user_channel_gain in user_channel_gains:
            user_channel_gains_normalized.append(interp(user_channel_gain,[0,5],[0,0.005]))

        user_features = [user_ids, user_distances_normalized, user_channel_gains_normalized]
        user_features = np.array(user_features).transpose()
        user_features_for_inference = []

        for user_feature in user_features:
            for feature in user_feature:
                user_features_for_inference.append(feature)
        user_features_for_inference = np.array(user_features_for_inference)
        #print('user_features')
        #print(user_features_for_inference)
        return user_features_for_inference

    def predict_future_association(self, access_point_radius, timestep_counter):
        preprocessed_inputs = self.preprocess_model_inputs(access_point_radius)
        preprocessed_inputs_tensor = torch.Tensor(preprocessed_inputs).to(self.device)
        association_prediction = self.access_point_model(preprocessed_inputs_tensor)
        association_prediction = association_prediction.detach().numpy()
        if timestep_counter < 5000:
            association_prediction = (association_prediction + np.random.normal(0, 0.2))

        #elif timestep_counter >= 1000:
            #association_prediction = (association_prediction + np.random.normal(0, 0.1))


        associations_prediction_mapped = []
        for prediction in association_prediction:
            associations_prediction_mapped.append(round(interp(prediction,[0,1],[1,self.num_access_points])))
        #print('associations_prediction_mapped')
        #print(associations_prediction_mapped)

        associations_prediction_mapped_for_global_model = copy.deepcopy(associations_prediction_mapped)
        associated_users_ids = []

        # print('SBS: ', self.SBS_label)
        # print('associations_prediction_mapped_for_global_model: ', associations_prediction_mapped_for_global_model)
        # print('association_prediction: ', association_prediction)

        # print('')

        associations = []
    
        for user in self.users:
            user_access_points_in_radius = []
            for x in user.access_points_within_radius:
                user_access_points_in_radius.append(x[0])
            if associations_prediction_mapped[user.user_label-1] not in user_access_points_in_radius:
                associations.append((user.user_label,self.SBS_label))
                association_prediction[user.user_label-1] = interp(self.SBS_label,[1,self.num_access_points],[0,1])
                associations_prediction_mapped[user.user_label-1] = self.SBS_label
            #else:
                #association_prediction.append((user.user_label, association_prediction[user.user_label-1]))

        for user in self.users:
            associated_users_ids.append(user.user_label)

        for user in self.all_users:
            if user.user_label not in associated_users_ids:
                associations_prediction_mapped_for_global_model[user.user_label-1] = 0
                association_prediction[user.user_label-1] = 0

        associations_prediction_mapped = np.array(associations_prediction_mapped)
        self.buffer_memory.append((preprocessed_inputs, association_prediction, 0))
        return associations_prediction_mapped_for_global_model
    
    def populate_buffer_memory_sample_with_reward(self,global_reward):
        rewards_in_memory = []
        if len(self.buffer_memory) > 1:
            new_sample = (self.buffer_memory[0][0],self.buffer_memory[0][1],global_reward)
            self.buffer_memory[0] = new_sample
            dnn_memory_rewards = []
            for sample in self.training_memory.storage:
                dnn_memory_rewards.append(sample[2])
            max_index = dnn_memory_rewards.index(max(dnn_memory_rewards))

            if global_reward >= dnn_memory_rewards[max_index]:
                self.training_memory.add(self.buffer_memory[0])
                #print('SBS: ', self.SBS_label, 'Appended')

            self.buffer_memory.pop(0)
        for sample in self.training_memory.storage:
            rewards_in_memory.append(sample[2])
        
        self.average_reward_in_memory = sum(rewards_in_memory)/len(rewards_in_memory)
        #print('SBS: ', self.SBS_label, self.training_memory.storage)

    def collect_state_space(self, eMBB_Users,urllc_users, Communication_Channel_1):
        total_gain = np.zeros(Communication_Channel_1.num_allocate_RBs_upper_bound*2)
        self.system_state_space_RB_channel_gains.clear()
        self.system_state_space_battery_energies.clear()
        channel_gains = []
        communication_queue_size = []
        battery_energy = []
        offloading_queue_lengths = []
        local_queue_lengths = []
        num_arriving_urllc_packets = []
        latency_requirement = []
        local_frequencies = []
        associated_embb_user_labels = []

        all_embb_users = []
        for user in self.all_users:
            if user.type_of_user_id == 0:
                all_embb_users.append(user)
        for embb_user in eMBB_Users:
            associated_embb_user_labels.append(embb_user.user_label)
        #reliability_requirement = []
        #Collect Channel gains\
        self.count_num_arriving_urllc_packets(urllc_users)
        for user in all_embb_users:
            if user.user_label in associated_embb_user_labels:
                channel_gains.append(embb_user.user_state_space.channel_gain)
                battery_energy.append(embb_user.user_state_space.battery_energy)
                offloading_queue_lengths.append(embb_user.user_state_space.offloading_queue_length)
                local_queue_lengths.append(embb_user.user_state_space.local_queue_length)
                num_arriving_urllc_packets.append(self.num_arriving_urllc_packets)
            else:
                channel_gains.append(total_gain)
                battery_energy.append(0)
                offloading_queue_lengths.append(0)
                local_queue_lengths.append(0)
                num_arriving_urllc_packets.append(self.num_arriving_urllc_packets)

        self.system_state_space_RB_channel_gains.append(channel_gains)
        self.system_state_space_battery_energies.append(battery_energy)
        return channel_gains, battery_energy, offloading_queue_lengths, local_queue_lengths, num_arriving_urllc_packets
        #return channel_gains, battery_energy

    def allocate_transmit_powers(self,eMBB_Users, action):
        index = 0
        # print('action')
        # print(action)
        for User in eMBB_Users:
            User.assigned_transmit_power_dBm = action[User.user_label-1]
            User.calculate_assigned_transmit_power_W()
            index+=1

    def allocate_offlaoding_ratios(self,eMBB_Users, action):
        index = 0
        for eMBB_User in eMBB_Users:
            eMBB_User.allocated_offloading_ratio = action[eMBB_User.user_label-1]
            index+=1

    def count_num_arriving_URLLC_packet(self,URLLC_Users):
        self.num_arriving_URLLC_packets = 0

        for URLLC_User in URLLC_Users:
            if URLLC_User.has_transmitted_this_time_slot == True:
                self.num_arriving_URLLC_packets += 1

    def receive_offload_packets(self, eMBB_Users):
        pass
        #for eMBB_User in eMBB_Users:
            #if eMBB_User.has_transmitted_this_time_slot == True:
                #self.eMBB_Users_packet_queue.append(eMBB_User.offloaded_packet)

    def calculate_achieved_total_system_energy_consumption(self, eMBB_Users):
        self.total_system_energy_consumption = 0
        for eMBB_User in eMBB_Users:
            self.achieved_total_system_energy_consumption += eMBB_User.achieved_total_energy_consumption

        #print("achieved_total_system_energy_consumption", self.achieved_total_system_energy_consumption)

    def calculate_achieved_total_system_processing_delay(self, eMBB_Users):
        self.achieved_total_system_processing_delay = 0
        for eMBB_User in eMBB_Users:
            self.achieved_total_system_processing_delay += eMBB_User.achieved_total_processing_delay

    def calculate_achieved_total_rate_URLLC_users(self, URLLC_Users):
        self.achieved_total_rate_URLLC_users = 0
        for URLLC_User in URLLC_Users:
            if URLLC_User.has_transmitted_this_time_slot == True:
                self.achieved_total_rate_URLLC_users += URLLC_User.achieved_channel_rate

    def calculate_achieved_total_rate_eMBB_users(self, eMBB_Users):
        self.achieved_total_rate_eMBB_users = 0
        for eMBB_User in eMBB_Users:
            if eMBB_User.has_transmitted_this_time_slot == True:
                self.achieved_total_rate_eMBB_users += eMBB_User.achieved_channel_rate

    def calculate_achieved_system_energy_efficiency(self):
        if self.achieved_total_system_energy_consumption == 0:
            self.achieved_system_energy_efficiency = 0
        else:
            self.achieved_system_energy_efficiency = self.achieved_total_rate_eMBB_users/self.achieved_total_system_energy_consumption

        #print("self.achieved_system_energy_efficiency",self.achieved_system_energy_efficiency)

    def calculate_achieved_system_reward(self, eMBB_Users, urllc_users, communication_channel):
        #print('number of embb users: ', len(eMBB_Users))
        self.achieved_system_reward = 0
        eMBB_User_energy_consumption = 0
        eMBB_User_channel_rate = 0
        eMBB_User_QOS_requirement_revenue_or_penelaty = 0
        total_energy = 0
        total_rate = 0
        total_QOS_revenue = 0
        self.fairness_index = 0
        individual_rewards = 0
        queue_delay_reward = 0
        delay = 0
        throughput_reward = 0
        resource_allocation_reward = 0
        tasks_dropped = 0

        self.individual_rewards.clear()
        self.individual_energy_rewards = []
        self.individual_channel_rate_rewards = []
        self.individual_channel_battery_energy_rewards = []
        self.individual_delay_rewards = []
        self.individual_queue_delays = []
        self.individual_tasks_dropped = []
        self.individual_energy_efficiency = []
        self.individual_total_reward = []
        total_users_energy_reward = 0
        total_users_throughput_reward = 0
        total_users_battery_energies_reward = 0
        total_users_delay_rewards = 0
        total_users_delay_times_energy_reward = 0
        total_users_resource_allocation_reward = 0
        overall_users_reward = 0
        total_eMBB_User_delay_normalized = 0
        total_offload_traffic_reward = 0
        total_lc_delay_violation_probability = 0
        urllc_reliability_reward, urllc_reliability_reward_normalized = self.calculate_urllc_reliability_reward(urllc_users)
        self.urllc_reliability_reward_normalized = urllc_reliability_reward_normalized
        for eMBB_User in eMBB_Users:
            eMBB_User_delay, eMBB_User_delay_normalized = eMBB_User.new_time_delay_calculation()
            eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption_normalized 
            total_energy += eMBB_User_energy_consumption
            eMBB_User_channel_rate = eMBB_User.achieved_channel_rate_normalized
            total_rate += eMBB_User_channel_rate
            delay_reward = eMBB_User.calculate_delay_penalty()
            battery_energy_reward = eMBB_User.energy_consumption_reward()
            energy_efficiency_reward = eMBB_User.calculate_energy_efficiency()
            #print('SBS: ', self.SBS_label, 'embb user: ',eMBB_User.user_label, 'energy efficiency: ', energy_efficiency_reward )
            resource_allocation_reward = eMBB_User.calculate_resource_allocation_reward(communication_channel)
            queue_delay_reward,delay = eMBB_User.calculate_queuing_delays()
            tasks_dropped = eMBB_User.tasks_dropped
            total_offload_traffic_reward += eMBB_User.offloading_queue_stability_constraint_reward()
            #total_lc_delay_violation_probability+=eMBB_User.local_queue_violation_constraint_reward()

            total_eMBB_User_delay_normalized+=eMBB_User_delay_normalized
            total_users_energy_reward += eMBB_User_energy_consumption
            total_users_throughput_reward += eMBB_User_channel_rate
            total_users_battery_energies_reward += battery_energy_reward
            total_users_delay_rewards += queue_delay_reward
            if eMBB_User_energy_consumption == 0:
                total_users_delay_times_energy_reward = 0
            else:
                total_users_delay_times_energy_reward += (queue_delay_reward*(1/eMBB_User_energy_consumption))
            total_users_resource_allocation_reward += resource_allocation_reward

        
            #if eMBB_User_energy_consumption == 0:
            #    individual_reward = 0
            #else:
            individual_reward = energy_efficiency_reward#*queue_delay_reward + battery_energy_reward  
      
            self.achieved_system_reward += individual_reward
            self.individual_rewards.append(individual_reward)

            self.energy_efficiency_rewards+=energy_efficiency_reward
            self.battery_energy_rewards+=battery_energy_reward
            self.throughput_rewards+=total_rate
            self.energy_rewards+=total_energy
            self.delay_rewards+=queue_delay_reward
            self.delays+=delay
            self.tasks_dropped+=tasks_dropped
            self.resource_allocation_rewards += resource_allocation_reward
            self.individual_energy_rewards.append(eMBB_User_energy_consumption)
            self.individual_channel_rate_rewards.append(eMBB_User_channel_rate)
            self.individual_energy_efficiency.append(energy_efficiency_reward)
            self.individual_total_reward.append(individual_reward)
            self.individual_channel_battery_energy_rewards.append(battery_energy_reward)
            self.individual_tasks_dropped.append(tasks_dropped)
            self.individual_delay_rewards.append(queue_delay_reward)
            self.individual_queue_delays.append(delay)
            self.total_reward += energy_efficiency_reward*queue_delay_reward + battery_energy_reward
            self.user_association_channel_rate_reward+=eMBB_User.calculate_achieved_user_association_channel_rate(communication_channel)

        for urllc_user in urllc_users:
            self.user_association_channel_rate_reward+=urllc_user.calculate_achieved_user_association_channel_rate()

        self.overall_users_reward = total_users_throughput_reward*total_users_delay_times_energy_reward + total_users_battery_energies_reward
        #overall_users_rewards = [overall_users_reward for _ in range(len(eMBB_Users))]
        #self.achieved_system_reward += urllc_reliability_reward_normalized
        fairness_index = self.calculate_fairness(eMBB_Users)
        #print('fairness index: ', fairness_index)
        fairness_index_normalized = 0.2*interp(fairness_index,[0,1],[0,1])
        #print('fairness index: ', fairness_index_normalized)
        #print(' ')
        #fairness_penalty = self.calculate_fairness_(eMBB_Users, communication_channel)
        #print('fairness penalty: ', fairness_penalty)
        #print('fairness penalty: ', fairness_index_normalized)
        #print(' ')
        self.fairness_index = fairness_index
        new_individual_rewards = [x + fairness_index_normalized for x in self.individual_rewards]
        #print('individual rewards: ', new_individual_rewards)
        #print('new individual rewards: ', new_inidividual_rewards)
        #print(' ')
        #print(' ')
        #print('new rewards')
        #new_inidividual_rewards = [fairness_index_normalized for _ in range(len(eMBB_Users))]
        #print(new_inidividual_rewards)

        #print("total_energy: ", total_energy)
        #print("total_rate: ", total_rate)
        #print("total_QOS_revenue: ", total_QOS_revenue)
        #self.achieved_system_reward
        return self.achieved_system_reward, self.achieved_system_reward, self.energy_rewards,self.throughput_rewards, self.user_association_channel_rate_reward
        #return self.achieved_system_reward, overall_users_rewards , self.energy_rewards,self.throughput_rewards

    def achieved_eMBB_delay_requirement_revenue_or_penalty(self,eMBB_User):
        processing_delay_requirement = eMBB_User.QOS_requirement_for_transmission.max_allowable_latency
        achieved_local_processing_delay = eMBB_User.achieved_local_processing_delay
        achieved_offload_processing_delay = eMBB_User.achieved_transmission_delay

        if processing_delay_requirement - max(achieved_local_processing_delay,achieved_offload_processing_delay) >= 0:
            return self.eMBB_User_delay_requirement_revenue
        else:
            return (processing_delay_requirement - max(achieved_local_processing_delay,achieved_offload_processing_delay))
        
    def achieved_URLLC_User_reliability_requirement_revenue_or_penalty(self,URLLC_User):
        reliability_requirement = URLLC_User.QOS_requirement_for_transmission.max_allowable_reliability
        achieved_reliability = self.achieved_total_rate_URLLC_users

        if ((achieved_reliability-reliability_requirement)/self.num_arriving_URLLC_packets) >= 0:
            return self.eMBB_User_delay_requirement_revenue
        else:
            return ((achieved_reliability-reliability_requirement)/self.num_arriving_URLLC_packets)
        
    #def perform_timeslot_sequential_events(self,eMBB_Users,URLLC_Users,communication_channel):

    def set_properties(self):
        self.user_association_channel_rate_reward = 0
        self.distance_exponent = -5
        self.associated_users = []
        self.associated_URLLC_users = []
        self.associated_eMBB_users = []
        self.system_state_space_RB_channel_gains = []
        self.system_state_space_battery_energies = []
        self.eMBB_Users_packet_queue = []
        self.URLLC_Users_packet_queue = []
        self.achieved_total_system_energy_consumption = 0
        self.achieved_total_system_processing_delay = 0
        self.achieved_URLLC_reliability = 0
        self.achieved_total_rate_URLLC_users = 0
        self.achieved_total_rate_eMBB_users = 0
        self.achieved_system_energy_efficiency = 0
        self.achieved_system_reward = 0
        self.eMBB_User_delay_requirement_revenue = 5
        self.URLLC_User_reliability_requirement_revenue = 5
        self.clock_frequency = 2
        self.work_load = 0
        #self.eMBB_UEs = []
        #self.URLLC_UEs = []
        self.achieved_users_energy_consumption = 0
        self.achieved_users_channel_rate = 0
        self.fairness_index = 0
        self.energy_efficiency_rewards = 0
        self.energy_rewards = 0
        self.throughput_rewards = 0
        self.delay_rewards = 0
        self.battery_energy_rewards = 0
        self.delays = 0
        self.tasks_dropped = 0
        self.resource_allocation_rewards = 0
        self.delay_reward_times_energy_reward = 0
        self.available_resource_time_blocks = []
        self.num_arriving_urllc_packets = 0
        self.urllc_reliability_constraint_max = 0.04
        self.K_mean = 0
        self.K_variance = 3
        self.outage_probability = 0
        self.previous_rates = []
        self.timeslot_counter = 0
        self.ptr = 0
        self.urllc_reliability_reward_normalized = 0
        self.q = 0
        self.individual_energy_rewards = []
        self.individual_channel_rate_rewards = []
        self.individual_channel_battery_energy_rewards = []
        self.individual_delay_rewards = []
        self.individual_queue_delays = []
        self.individual_tasks_dropped = []
        self.individual_energy_efficiency = []
        self.individual_total_reward = []
        self.total_reward = 0
        self.overall_users_reward = []

    def calculate_fairness(self,eMBB_Users):
        number_of_users = len(eMBB_Users)
        sum_throughputs = 0
        for eMBB_User in eMBB_Users:
            sum_throughputs += eMBB_User.achieved_channel_rate

        square_sum_throughput = math.pow(sum_throughputs,2)
        sum_square_throughput = self.square_sum(eMBB_Users)
        fairness_index = 0
        if sum_square_throughput > 0:
            fairness_index = square_sum_throughput/(number_of_users*sum_square_throughput)

        return fairness_index

    def square_sum(self,eMBB_Users):
        sum = 0
        for eMBB_User in eMBB_Users:
            sum+=math.pow(eMBB_User.achieved_channel_rate,2)

        return sum
    
    def calculate_fairness_(self,eMBB_Users, communication_channel):
        sum_square_error = 0
        for eMBB_user in eMBB_Users:
            square_error = math.pow(abs(len(eMBB_user.allocated_RBs)-communication_channel.num_of_RBs_per_User),2)
            sum_square_error+=square_error

        sum_square_error = math.pow(abs((len(eMBB_Users[0].allocated_RBs))-(len(eMBB_Users[1].allocated_RBs))),2)
        if sum_square_error == 0:
            sum_square_error = 1
        
        return 1/sum_square_error
    
    def allocate_resource_blocks_URLLC(self,communication_channel, URLLC_Users):
        for URLLC_user in URLLC_Users:
            URLLC_user.calculate_channel_gain_on_all_resource_blocks(communication_channel)

        for rb in range(1,communication_channel.num_allocate_RBs_upper_bound+1):
            for tb in range(1,communication_channel.time_divisions_per_slot+1):
                self.available_resource_time_blocks.append((tb,rb))
        
        for urllc_user in URLLC_Users:
            random_number = np.random.randint(0, len(self.available_resource_time_blocks), 1)
            random_number = random_number[0]
            urllc_user.assigned_resource_time_block = self.available_resource_time_blocks[random_number]
            self.available_resource_time_blocks = np.delete(self.available_resource_time_blocks,random_number,axis=0)
            urllc_user.assigned_time_block = urllc_user.assigned_resource_time_block[0]
            urllc_user.assigned_resource_block = urllc_user.assigned_resource_time_block[1]

    def count_num_arriving_urllc_packets(self, urllc_users):
        self.num_arriving_urllc_packets = 0
        for urllc_user in urllc_users:
            if urllc_user.has_transmitted_this_time_slot == True:
                self.num_arriving_urllc_packets += 1


        

        

        # for urllc_user in URLLC_Users:
        #     print('urllc user: ', urllc_user.URLLC_UE_label, 'allocated resource block id: ', urllc_user.assigned_resource_block)
        #     print('urllc user: ', urllc_user.URLLC_UE_label, 'allocated time block id: ', urllc_user.assigned_time_block)

        # print('')

        


        



        # URLLC_Users_temp = URLLC_Users
        # RB_URLLC_mapping = []
        # for x in range(1,communication_channel.num_allocate_RBs_upper_bound+1):
        #     for y in range(0,communication_channel.num_urllc_users_per_RB):
        #         if len(URLLC_Users_temp) > 0:

    def get_top_two_indices(arr):
        # Get the indices of the array sorted in ascending order
        sorted_indices = np.argsort(arr)

        # Get the index of the largest number (last index after sorting)
        largest_index = sorted_indices[-1]

        # Get the index of the second largest number (second-to-last index after sorting)
        second_largest_index = sorted_indices[-2]

        return largest_index
    
    def calculate_urllc_reliability_reward(self, urllc_users):
        num_arriving_urllc_packets = self.num_arriving_urllc_packets
        urllc_task_size = 0
        if len(urllc_users) > 0:
            urllc_task_size = urllc_users[0].task_size_per_slot_bits    

        urllc_total_rate = 0
        for urllc_user in urllc_users:
            urllc_total_rate+=urllc_user.achieved_channel_rate

      
       
        K = num_arriving_urllc_packets*urllc_task_size
        K_mean = (len(urllc_users)/2)*urllc_task_size
        K_variance = self.K_variance*urllc_task_size
        K_inv = stats.norm.ppf((1-self.urllc_reliability_constraint_max), loc=K_mean, scale=K_variance)
        #print('K_inv: ', K_inv)
        #print('urllc_total_rate: ', urllc_total_rate)
        # K = num_arriving_urllc_packets*urllc_task_size
        # self.K_mean = (len(urllc_users)/2)*urllc_task_size
        # #K_cdf = stats.norm.cdf(K,self.K_mean,self.K_variance)
        # K_cdf = stats.norm.cdf(K,self.K_mean,self.K_variance)
        # self.outage_probability = 1 - K_cdf

        # print('self.K_mean: ', self.K_mean)
        #print('self.K-cdf: ', K_cdf)
        # print('self.1/K-cdf: ', 1/K_cdf)
        # print('total urllc rate: ', urllc_total_rate)
        # print('(1/K_cdf)*(1-self.urllc_reliability_constraint_max): ', (1/K_cdf)*(1-self.urllc_reliability_constraint_max))
        #print('urllc rate: ', urllc_total_rate)
        reliability_reward = urllc_total_rate-K_inv
        average_rate_prev_slots = self.urllc_rate_expectation_over_prev_T_slot(10,urllc_total_rate)
        #print('self.previous_rates: ', self.previous_rates)
        variance = urllc_task_size

        self.outage_probability = stats.norm.cdf(K,loc=average_rate_prev_slots,scale=variance)
        # print('reliability_reward: ', reliability_reward)
        # print('self.outage_probability: ', self.outage_probability)
        #print('reliability_reward: ', reliability_reward)
        reliability_reward_max = 2000
        reliability_reward_min = -2000
        reliability_reward_normalized = interp(reliability_reward,[reliability_reward_min,reliability_reward_max],[0,5])
        return reliability_reward, reliability_reward_normalized
    
    def urllc_rate_expectation_over_prev_T_slot(self, T, urllc_total_rate):
        self.timeslot_counter+=1
        number_of_previous_time_slots = T

        if len(self.previous_rates) == number_of_previous_time_slots:
            self.previous_rates[int(self.ptr)] = urllc_total_rate
            self.ptr = (self.ptr + 1) % number_of_previous_time_slots
        else:
            self.previous_rates.append(urllc_total_rate)

        average_rate = sum(self.previous_rates)/len(self.previous_rates)
        return average_rate
    

    def reward(self, eMBB_Users, urllc_users, communication_channel):
        #print('number of embb users: ', len(eMBB_Users))
        self.achieved_system_reward = 0
        eMBB_User_energy_consumption = 0
        eMBB_User_channel_rate = 0
        eMBB_User_QOS_requirement_revenue_or_penelaty = 0
        total_energy = 0
        total_rate = 0
        total_QOS_revenue = 0
        self.fairness_index = 0
        individual_rewards = 0
        queue_delay_reward = 0
        delay = 0
        throughput_reward = 0
        resource_allocation_reward = 0
        tasks_dropped = 0

        self.individual_rewards.clear()

        total_users_energy_reward = 0
        total_users_throughput_reward = 0
        total_users_battery_energies_reward = 0
        total_users_delay_rewards = 0
        total_users_delay_times_energy_reward = 0
        total_users_resource_allocation_reward = 0
        overall_users_reward = 0
        total_eMBB_User_delay_normalized = 0
        total_offload_traffic_reward = 0
        total_lc_delay_violation_probability = 0
        urllc_reliability_reward, urllc_reliability_reward_normalized = self.calculate_urllc_reliability_reward(urllc_users)
        self.urllc_reliability_reward_normalized = urllc_reliability_reward_normalized
        for eMBB_User in eMBB_Users:
            eMBB_User_delay, eMBB_User_delay_normalized = eMBB_User.new_time_delay_calculation()
            eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption 
            total_energy += eMBB_User_energy_consumption
            eMBB_User_channel_rate = eMBB_User.achieved_channel_rate
            total_rate += eMBB_User_channel_rate
            delay_reward = eMBB_User.calculate_delay_penalty()
            battery_energy_reward = eMBB_User.energy_consumption_reward()
            energy_efficiency_reward = eMBB_User.calculate_energy_efficiency()
            resource_allocation_reward = eMBB_User.calculate_resource_allocation_reward(communication_channel)
            queue_delay_reward,delay = eMBB_User.calculate_queuing_delays()
            tasks_dropped = eMBB_User.tasks_dropped
            total_offload_traffic_reward += eMBB_User.offloading_queue_stability_constraint_reward()
            total_lc_delay_violation_probability+=eMBB_User.local_queue_violation_constraint_reward()

            total_eMBB_User_delay_normalized+=eMBB_User_delay_normalized
            total_users_energy_reward += eMBB_User_energy_consumption
            total_users_throughput_reward += eMBB_User_channel_rate
            total_users_battery_energies_reward += battery_energy_reward
            total_users_delay_rewards += queue_delay_reward
            if eMBB_User_energy_consumption == 0:
                total_users_delay_times_energy_reward = 0
            else:
                total_users_delay_times_energy_reward += (queue_delay_reward*(1/eMBB_User_energy_consumption))
            total_users_resource_allocation_reward += resource_allocation_reward

        
            #if eMBB_User_energy_consumption == 0:
            #    individual_reward = 0
            #else:
            individual_reward = energy_efficiency_reward#*queue_delay_reward + battery_energy_reward  
      
            self.achieved_system_reward += individual_reward
            self.individual_rewards.append(individual_reward)

            self.energy_efficiency_rewards+=energy_efficiency_reward
            self.battery_energy_rewards+=battery_energy_reward
            self.throughput_rewards+=total_rate
            self.energy_rewards+=total_energy
            self.delay_rewards+=queue_delay_reward
            self.delays+=delay
            self.tasks_dropped+=tasks_dropped
            self.resource_allocation_rewards += resource_allocation_reward

        
        self.q = total_users_throughput_reward*total_users_delay_times_energy_reward 
        reward = total_users_throughput_reward - self.q*total_users_delay_times_energy_reward + total_users_battery_energies_reward + urllc_reliability_reward + total_offload_traffic_reward +total_lc_delay_violation_probability
        
        return self.achieved_system_reward, reward , self.energy_rewards,self.throughput_rewards





        







