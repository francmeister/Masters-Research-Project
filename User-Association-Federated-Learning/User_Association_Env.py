import gym
from gym import spaces
import pygame, sys, time, random, numpy as np
from eMBB_UE import eMBB_UE
from URLLC_UE import URLLC_UE
from Communication_Channel import Communication_Channel
from SBS import SBS
from numpy import interp
import pandas as pd
import copy

pygame.init()

#Set constant variables
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900
ENV_WIDTH_PIXELS = 1100
ENV_HEIGHT_PIXELS = 900
ENV_WIDTH_METRES = 400
ENV_HEIGHT_METRES = 400

clock = pygame.time.Clock()
#screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

class NetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, all_users):
        self.number_of_users = len(all_users)
        

        action_space_high = np.array([1 for _ in range(self.number_of_users)], dtype=np.float32)

       
        action_space_low = np.array([0 for _ in range(self.number_of_users)],dtype=np.float32)
        action_space_high = np.transpose(action_space_high)
        action_space_low = np.transpose(action_space_low)
        
        observation_space_high = np.array([0 for _ in range(self.number_of_users)], dtype=np.float32)

        observation_space_low = np.array([0 for _ in range(self.number_of_users)], dtype=np.float32)


    
     
        self.box_action_space = spaces.Box(low=action_space_low,high=action_space_high)
        self.number_of_box_actions = 2
        self.box_action_space_len = 0
        self.binary_action_space = spaces.MultiBinary(self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
        self.binary_action_space_len = 0

        # Combine the action spaces into a dictionary
        #self.action_space = self.box_action_space
        
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high)

        #self.action_space = spaces.Box(low=action_space_low,high=action_space_high)
        self.observation_space = spaces.Box(low=observation_space_low, high=observation_space_high) 
      
        self.action_space_high = 1
        self.action_space_low = 0

        self.STEP_LIMIT = 30
        self.sleep = 0
        self.steps = 0
        self.episode_reward = 0


            
    def step(self,action):
        

    
        #print('self.offload decisions')
        #print(offload_decision_mapped)

        #Perform Actions
        self.SBS.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions_mapped)
        #self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions)

        self.SBS.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions_mapped)
        #self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions)

        #self.Communication_Channel_1.number_URLLC_Users_per_RB = number_URLLC_Users_per_RB_action_mapped
        #self.Communication_Channel_1.number_URLLC_Users_per_RB = number_URLLC_Users_per_RB_action

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS)
        self.Communication_Channel_1.initiate_RBs()
        
        self.Communication_Channel_1.allocate_RBs_eMBB(self.eMBB_Users,RB_allocation_actions)
        #self.Communication_Channel_1.allocate_subcarriers_eMBB(self.eMBB_Users,subcarrier_allocation_actions)
        #self.Communication_Channel_1.create_resource_blocks_URLLC()
        #self.Communication_Channel_1.allocate_resource_blocks_URLLC(self.URLLC_Users)
        #self.Communication_Channel_1.subcarrier_URLLC_User_mapping()


        for eMBB_User in self.eMBB_Users:
            eMBB_User.increment_task_queue_timers()
            eMBB_User.split_tasks()
       

        for eMBB_User in self.eMBB_Users:
            #if eMBB_User.has_transmitted_this_time_slot == True:
            eMBB_User.transmit_to_SBS(self.Communication_Channel_1,self.URLLC_Users)
            #print('eMBB_User id: ', eMBB_User.eMBB_UE_label, 'achieved channel rate: ', eMBB_User.achieved_channel_rate)
            eMBB_User.local_processing()
            eMBB_User.offloading(self.Communication_Channel_1)
            eMBB_User.total_energy_consumed()
            eMBB_User.total_processing_delay()

        #print('')
        for URLLC_user in self.URLLC_Users:
            URLLC_user.calculate_achieved_channel_rate(self.eMBB_Users,self.Communication_Channel_1)
            #print('urllc user id: ', URLLC_user.URLLC_UE_label, 'achieved channel rate: ', URLLC_user.achieved_channel_rate)

        #print('')
       

        self.SBS.receive_offload_packets(self.eMBB_Users)
        self.SBS.calculate_achieved_total_system_energy_consumption(self.eMBB_Users)
        self.SBS.calculate_achieved_total_system_processing_delay(self.eMBB_Users)
        self.SBS.calculate_achieved_total_rate_eMBB_users(self.eMBB_Users)
        self.SBS.calculate_achieved_system_energy_efficiency()
        system_reward, reward, self.total_energy,self.total_rate,user_association_channel_rate_reward = self.SBS.calculate_achieved_system_reward(self.eMBB_Users,self.URLLC_Users,self.Communication_Channel_1)
        self.user_association_channel_rate_reward+=user_association_channel_rate_reward
        #reward = [x + resource_block_allocation_penalty for x in reward]
       
        
        #print('Reward')
        #print(reward)
        #print(' ')
        #mapped_reward = interp(reward,[0,1000],[7200000000,7830000000])
        #Update game state after performing actions
        for eMBB_User in self.eMBB_Users:
            eMBB_User.calculate_distance_from_SBS(self.SBS.x_position, self.SBS.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
            eMBB_User.calculate_channel_gain(self.Communication_Channel_1)
            eMBB_User.calculate_user_association_channel_gains()
            eMBB_User.harvest_energy()
            eMBB_User.compute_battery_energy_level()
            eMBB_User.generate_task(self.Communication_Channel_1)
            eMBB_User.collect_state()

        for urllc_user in self.URLLC_Users:
            urllc_user.calculate_channel_gain_on_all_resource_blocks(self.Communication_Channel_1)
            urllc_user.calculate_user_association_channel_gains()
            urllc_user.generate_task(self.Communication_Channel_1)
            urllc_user.split_tasks()

        observation_channel_gains, observation_battery_energies, observation_offloading_queue_lengths, observation_local_queue_lengths, num_urllc_arriving_packets = self.SBS.collect_state_space(self.eMBB_Users, self.URLLC_Users,  self.Communication_Channel_1)
        
        #observation_channel_gains, observation_battery_energies = self.SBS1.collect_state_space(self.eMBB_Users)
        #observation_channel_gains = np.array(observation_channel_gains, dtype=np.float32)
        #observation_battery_energies = np.array(observation_battery_energies, dtype=np.float32)
        #print('Observation before transpose')
        #print(np.transpose(observation))
        #normalize observation values to a range between 0 and 1 using interpolation
        row = 0
        for num_urllc_arriving_packet in num_urllc_arriving_packets:
            num_urllc_arriving_packets[row] = interp(num_urllc_arriving_packets[row],[0,len(self.URLLC_Users)],[0,1])
            row+=1
        row = 0
        col = 0
        min_value = 0
        max_value = 0
        for channel_gains in observation_channel_gains:
            for channel_gain in channel_gains:
                observation_channel_gains[row][col] = interp(observation_channel_gains[row][col],[self.channel_gain_min,self.channel_gain_max],[0,1])
                col+=1

            row+=1
            col = 0

        row = 0
        for battery_energy in observation_battery_energies:
            observation_battery_energies[row] = interp(observation_battery_energies[row],[self.battery_energy_min,self.battery_energy_max],[0,1])
            row+=1
        
        row = 0
        for offloading_queue_length in observation_offloading_queue_lengths:
            observation_offloading_queue_lengths[row] = interp(observation_offloading_queue_lengths[row],[self.min_off_queue_length,self.max_off_queue_length],[0,1])
            row+=1

        row = 0
        for local_queue_length in observation_local_queue_lengths:
            observation_local_queue_lengths[row] = interp(observation_local_queue_lengths[row],[self.min_lc_queue_length,self.max_lc_queue_length],[0,1])
            row+=1
    
        observation_channel_gains = np.array(observation_channel_gains).squeeze()
        
        observation_battery_energies = np.array(observation_battery_energies)
        observation_offloading_queue_lengths = np.array(observation_offloading_queue_lengths)
        observation_local_queue_lengths = np.array(observation_local_queue_lengths)
        num_urllc_arriving_packets = np.array(num_urllc_arriving_packets)

        if self.number_of_users == 1:
            observation_channel_gains_num = len(observation_channel_gains)
            observation_battery_energies_num = len(observation_battery_energies)
            #observation_offloading_queue_lengths_num = len(observation_offloading_queue_lengths)
            #observation_local_queue_lengths_num = len(observation_local_queue_lengths)

            observation_channel_gains = observation_channel_gains.reshape(observation_battery_energies_num,observation_channel_gains_num)
        
      
        #observation_channel_gains = np.transpose(observation_channel_gains)
        #observation_battery_energies = np.transpose(observation_battery_energies)
        observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths,num_urllc_arriving_packets)) #observation_channel_gains.
        #print('observation matrix')
        observation = self.reshape_observation_space_for_model(observation)
       

        done = self.check_timestep()
        dones = [0 for element in range(len(self.eMBB_Users) - 1)]
        dones.append(done)
        info = {'reward': reward}
        self.steps+=1
        #print('Timestep: ', self.steps)
        #print('reward: ', reward)
        self.rewards.append(reward)
        self.episode_reward+=reward
        #print(' ')
        
        penalty_per_RB = -(1/self.num_allocate_RB_upper_bound)
        penalty_accumulation = 0
        sum_allocations_per_RB_matrix = np.sum(resource_block_action_matrix, axis=0)
        self.sum_allocations_per_RB_matrix = sum_allocations_per_RB_matrix
        #print(self.sum_allocations_per_RB_matrix)
        if not np.all(np.sum(resource_block_action_matrix, axis=0) <= 1):
      
            for sum_allocations_per_RB in sum_allocations_per_RB_matrix:
                if sum_allocations_per_RB >= 1:
                    penalty_accumulation += ((sum_allocations_per_RB-1)*penalty_per_RB)
                   
                elif sum_allocations_per_RB == 0:
                    penalty_accumulation += -0.2#((1-sum_allocations_per_RB)*penalty_per_RB)
                elif sum_allocations_per_RB == 1:
                    penalty_accumulation += 0.5

            

        #      #penalty_accumulation = interp(penalty_accumulation,[-1,0],[-1,5])

        elif np.all(np.sum(resource_block_action_matrix, axis=0) == 1):
            for x in range(0,self.num_allocate_RB_upper_bound):
                penalty_accumulation += 1

        for sum_allocations_per_RB in sum_allocations_per_RB_matrix:       
            if sum_allocations_per_RB == 0:
                penalty_accumulation += -0.2#((1-sum_allocations_per_RB)*penalty_per_RB)
        
        #print(penalty_accumulation)
        #row = 0
      
        #for item in reward:
            #if item > 0: 
        #    reward[row] = penalty_accumulation
        #    row+=1
        #dones[len(dones)-1] = 1
        #print(reward)
        #print('')
        return observation,reward,done,info
    
    
    
    def reset(self):
        self.user_association_channel_rate_reward = 0
        self.episode_reward = 0
        self.steps = 0
        self.SBS.set_properties()
        self.OS_channel_gain_label = 0
        self.OS_comm_queue_label = 1
        self.OS_latency_label = 3
        self.OS_battery_energy_label = 2
        self.OS_cpu_frequency_label = 4

        #Observation Space Bound Parameters
        self.channel_gain_min = self.eMBB_UE_1.min_channel_gain
        self.channel_gain_max = self.eMBB_UE_1.max_channel_gain
        self.communication_queue_min = self.eMBB_UE_1.min_communication_qeueu_size
        self.communication_queue_max = self.eMBB_UE_1.max_communication_qeueu_size
        self.battery_energy_min = 0
        self.battery_energy_max = self.eMBB_UE_1.max_battery_energy
        self.latency_requirement_min = 0
        self.latency_requirement_max = self.eMBB_UE_1.max_allowable_latency
        self.cpu_frequency_max = self.eMBB_UE_1.max_cpu_frequency
        self.cpu_frequency_min = self.eMBB_UE_1.min_cpu_frequency
        self.max_lc_queue_length = self.eMBB_UE_1.max_lc_queue_length
        self.min_lc_queue_length = 0
        self.max_off_queue_length = self.eMBB_UE_1.max_off_queue_length
        self.min_off_queue_length = 0
        self.resource_block_allocation_matrix = []
        self.resource_allocation_constraint_violation = 0
        self.eMBB_Users = copy.deepcopy(self.SBS.embb_users)
        self.URLLC_Users = copy.deepcopy(self.SBS.urllc_users)
        distances = []
        access_points = []
        users = []

        #print('SBS: ', self.SBS.SBS_label, 'Number of connected users: ', len(self.eMBB_Users))
        #print('SBS: ', self.SBS.SBS_label, 'Number of users: ', len(self.eMBB_Users)+len(self.URLLC_Users), 'embb users: ',len(self.eMBB_Users), 'urllc users: ', len(self.URLLC_Users))
       
        for eMBB_User in self.eMBB_Users:
            #eMBB_User.set_properties_UE()
            eMBB_User.set_properties_eMBB()
            eMBB_User.collect_state()
            eMBB_User.current_associated_access_point = self.SBS.SBS_label
            eMBB_User.calculate_distances_from_access_point(self.access_point_coordinates, self.radius)
            eMBB_User.calculate_user_association_channel_gains()
            eMBB_User.calculate_distance_from_current_access_point()
            distances.append(eMBB_User.distance_from_associated_access_point)
            access_points.append(eMBB_User.current_associated_access_point)
            users.append(eMBB_User.user_label)

        distances = np.array(distances)
        access_points = np.array(access_points)

        #print('SBS: ', self.SBS.SBS_label, 'Users: ', users, 'distances from associated access points: ', distances)
        #print('associated access points: ', access_points)
        # print('')
        # print('')
        #print('SBS: ', self.SBS.SBS_label, 'associated users: ', associated_users)

        for URLLC_User in self.URLLC_Users:
            URLLC_User.set_properties_UE()
            URLLC_User.set_properties_URLLC()
            URLLC_User.current_associated_access_point = self.SBS.SBS_label
            eMBB_User.calculate_distances_from_access_point(self.access_point_coordinates, self.radius)
            URLLC_User.calculate_user_association_channel_gains()
            URLLC_User.calculate_distance_from_current_access_point()

        #self.eMBB_Users.clear()
        #self.URLLC_Users.clear()
        #self.group_users()

        #self.SBS.associate_users(self.eMBB_Users, self.URLLC_Users)
        self.Communication_Channel_1.set_properties()

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS)
        self.Communication_Channel_1.initiate_RBs()
        self.SBS.allocate_resource_blocks_URLLC(self.Communication_Channel_1, self.URLLC_Users)
        
        info = {'reward': 0}
        #print('battery enegy: ', self.SBS1.system_state_space[4])
        #observation_channel_gains, observation_battery_energies = self.SBS1.collect_state_space(self.eMBB_Users)
        observation_channel_gains, observation_battery_energies, observation_offloading_queue_lengths, observation_local_queue_lengths, num_urllc_arriving_packets = self.SBS.collect_state_space(self.eMBB_Users, self.URLLC_Users, self.Communication_Channel_1)
        #observation_channel_gains = np.array(observation_channel_gains, dtype=np.float32)
        #observation_battery_energies = np.array(observation_battery_energies, dtype=np.float32)
        #print('Observation before transpose')
        #print(np.transpose(observation))
        #normalize observation values to a range between 0 and 1 using interpolation
        row=0
        for num_urllc_arriving_packet in num_urllc_arriving_packets:
            num_urllc_arriving_packets[row] = interp(num_urllc_arriving_packets[row],[0,len(self.URLLC_Users)],[0,1])
            row+=1

        row = 0
        col = 0
        min_value = 0
        max_value = 0
  
        for channel_gains in observation_channel_gains:
            for channel_gain in channel_gains:
                observation_channel_gains[row][col] = interp(observation_channel_gains[row][col],[self.channel_gain_min,self.channel_gain_max],[0,1])
                col+=1

            row+=1
            col = 0

        row = 0
        for battery_energy in observation_battery_energies:
            observation_battery_energies[row] = interp(observation_battery_energies[row],[self.battery_energy_min,self.battery_energy_max],[0,1])
            row+=1
        
        row = 0
        for offloading_queue_length in observation_offloading_queue_lengths:
            observation_offloading_queue_lengths[row] = interp(observation_offloading_queue_lengths[row],[self.min_off_queue_length,self.max_off_queue_length],[0,1])
            row+=1

        row = 0
        for local_queue_length in observation_local_queue_lengths:
            observation_local_queue_lengths[row] = interp(observation_local_queue_lengths[row],[self.min_lc_queue_length,self.max_lc_queue_length],[0,1])
            row+=1

        #observation_channel_gains = np.transpose(observation_channel_gains)
        #observation_battery_energies = np.transpose(observation_battery_energies)
        observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths,num_urllc_arriving_packets)) #observation_channel_gains.
        observation = self.reshape_observation_space_for_model(observation)
        reward = 0
        done = 0
        return observation
       
    def render(self, mode='human'):
        pass

    def create_objects(self, SBS):
        #Small Cell Base station
        self.SBS = SBS
        self.eMBB_Users = SBS.embb_users
        self.URLLC_Users = SBS.urllc_users
        
        #print('self.user_association_epoch_number: ', self.user_association_epoch_number)
        #print('access point id: ', self.access_point_id)
        print('embbusers: ', len(self.eMBB_Users))
        print('urllc users: ',len(self.URLLC_Users))
        print('')
        
        #Users
        self.eMBB_UE_1 = eMBB_UE(1,2,100,600)

        if len(self.eMBB_Users) == 0:
           self.eMBB_Users.append(self.eMBB_UE_1)
        # self.eMBB_UE_2 = eMBB_UE(2,100,600)
        # self.eMBB_UE_3 = eMBB_UE(3,100,600)

        # self.URLLC_UE_1 = URLLC_UE(1,100,600)
        # self.URLLC_UE_2 = URLLC_UE(2,100,600)
        # self.URLLC_UE_3 = URLLC_UE(3,100,600)
        # self.URLLC_UE_4 = URLLC_UE(4,100,600)
        # self.URLLC_UE_5 = URLLC_UE(5,100,600)
        # self.URLLC_UE_6 = URLLC_UE(6,100,600)


        #Communication Channel
        self.Communication_Channel_1 = Communication_Channel(self.SBS.SBS_label)

        #Group Users

        #self.group_users()

        #Associate SBS with users
        #self.SBS.associate_users(self.eMBB_Users,self.URLLC_Users)

    #def group_users(self):
        #Group all eMBB Users
        # self.eMBB_Users.append(self.eMBB_UE_1)
        # self.eMBB_Users.append(self.eMBB_UE_2)
        # self.eMBB_Users.append(self.eMBB_UE_3)

        # self.URLLC_Users.append(self.URLLC_UE_1)
        # self.URLLC_Users.append(self.URLLC_UE_2)
        # self.URLLC_Users.append(self.URLLC_UE_3)
        # self.URLLC_Users.append(self.URLLC_UE_4)
        # self.URLLC_Users.append(self.URLLC_UE_5)
        # self.URLLC_Users.append(self.URLLC_UE_6)

    def check_timestep(self):
        if self.steps >= self.STEP_LIMIT:
            return True
        else: 
            return False
        
    def seed(self):
        pass