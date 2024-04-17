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

class SBS():
    def __init__(self, SBS_label):
        self.SBS_label = SBS_label
        #SBS Telecom properties
        self.x_position = 200
        self.y_position = 200
        self.individual_rewards = []
        self.training_memory = DNN_TRAINING_MEMORY()
        #self.access_point_model = DNN(input_dim,output_dim)
        self.buffer_memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_properties()

    def get_all_users(self, all_users):
        self.all_users = all_users

    def associate_users(self, users):
        self.users = users
        self.embb_users = []
        self.urllc_users = []
        for user in users:
            if user.type_of_user_id == 0:
                self.embb_users.append(user)
            elif user.type_of_user_id == 1:
                self.urllc_users.append(user)

    def initialize_DNN_model(self,global_model):
        self.access_point_model = global_model

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
        self.access_point_model.load_state_dict(global_model.state_dict())
        #self.access_point_model.to(device)

    def acquire_global_memory(self, global_memory):    
        self.training_memory = global_memory   

    def train_local_dnn(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.access_point_model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.access_point_model.parameters(), lr=0.001)
        self.num_training_epochs = 500
        x_train, y_train, sample_rewards = self.training_memory.sample(20)
        # print('len(x_train[0]): ', y_train[0])
        #print(len(x_train[0]))

        x_train_tensor = torch.Tensor(x_train).to(device)
        y_train_tensor = torch.Tensor(y_train).to(device)

        if x_train_tensor.dtype != self.access_point_model.fc1.weight.dtype:
            x_train_tensor = x_train_tensor.to(self.access_point_model.fc1.weight.dtype)
            y_train_tensor = y_train_tensor.to(self.access_point_model.fc1.weight.dtype)

        print('Starting training of local DNN of Access Point: ', self.SBS_label)
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
            #print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('Finished training local DNN of Access Point: ', self.SBS_label)
        return self.access_point_model

        # y_pred = self.access_point_model(x_train_tensor)
        # print('y_train_tensor[0]')
        # print(y_train_tensor[0])

        # print('y_pred[0]')
        # print(y_pred[0])
    def predict_future_association(self):
        preprocessed_inputs = self.preprocess_model_inputs()
        preprocessed_inputs_tensor = torch.Tensor(preprocessed_inputs).to(self.device)
        association_prediction = self.access_point_model(preprocessed_inputs_tensor)
    #     #get the input data
    #     input_features = []
    #     for user in self.all_users:
    #         if user in self.users:
    #             input_features.append(user.user_label)
    #             input_features.append(user.distance_from_associated_access_point)
    #             input_features.append(user.user_association_channel_gain)
    #             input_features.append(user.)

    def collect_state_space(self, eMBB_Users,urllc_users):
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
        embb_user_labels = []
        for embb_user in eMBB_Users:
            embb_user_labels.append(embb_user.user_id)
        #reliability_requirement = []
        #Collect Channel gains
        self.count_num_arriving_urllc_packets(urllc_users)
        for user in self.all_users:
            if user.user_label in embb_user_labels:
                channel_gains.append(embb_user.user_state_space.channel_gain)
                battery_energy.append(embb_user.user_state_space.battery_energy)
                offloading_queue_lengths.append(embb_user.user_state_space.offloading_queue_length)
                local_queue_lengths.append(embb_user.user_state_space.local_queue_length)
                num_arriving_urllc_packets.append(self.num_arriving_urllc_packets)
            else:
                channel_gains.append(0)
                battery_energy.append(0)
                offloading_queue_lengths.append(0)
                local_queue_lengths.append(0)
                num_arriving_urllc_packets.append(0)

        self.system_state_space_RB_channel_gains.append(channel_gains)
        self.system_state_space_battery_energies.append(battery_energy)
        return channel_gains, battery_energy, offloading_queue_lengths, local_queue_lengths, num_arriving_urllc_packets
        #return channel_gains, battery_energy

    def allocate_transmit_powers(self,eMBB_Users, action):
        index = 0
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

    def calculate_achieved_system_reward(self, eMBB_Users, communication_channel):
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

        for eMBB_User in eMBB_Users:
            eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption_normalized 
            total_energy += eMBB_User_energy_consumption
            eMBB_User_channel_rate = eMBB_User.achieved_channel_rate_normalized
            total_rate += eMBB_User_channel_rate
            delay_reward = eMBB_User.calculate_delay_penalty()
            battery_energy_reward = eMBB_User.energy_consumption_reward()
            energy_efficiency_reward = eMBB_User.calculate_energy_efficiency()
            resource_allocation_reward = eMBB_User.calculate_resource_allocation_reward(communication_channel)
            queue_delay_reward,delay = eMBB_User.calculate_queuing_delays()
            tasks_dropped = eMBB_User.tasks_dropped

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
            individual_reward = energy_efficiency_reward#*queue_delay_reward #+ battery_energy_reward  
      
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

        #overall_users_reward = 1/total_users_energy_reward#total_users_throughput_reward*total_users_delay_times_energy_reward + total_users_battery_energies_reward
        #overall_users_rewards = [overall_users_reward for _ in range(len(eMBB_Users))]
       
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
  
        return self.achieved_system_reward, self.individual_rewards , self.energy_rewards,self.throughput_rewards
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
        self.eMBB_UEs = []
        self.URLLC_UEs = []
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




        







