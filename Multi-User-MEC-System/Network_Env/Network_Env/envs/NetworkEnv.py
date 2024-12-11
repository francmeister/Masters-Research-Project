import gym
from gym import spaces
import sys, time, random, numpy as np
from eMBB_UE import eMBB_UE
from URLLC_UE import URLLC_UE
from Communication_Channel import Communication_Channel
from SBS import SBS
from numpy import interp
import pandas as pd



#Set constant variables
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900
ENV_WIDTH_PIXELS = 1100
ENV_HEIGHT_PIXELS = 900
ENV_WIDTH_METRES = 400
ENV_HEIGHT_METRES = 400

#clock = pygame.time.Clock()
#screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

class NetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.timestep_counter = 0
        self.create_objects()
        self.reset()
        #Action Space Bound Paramaters
        #self.access_point_id = access_point_id
        self.max_offload_decision = 1
        self.min_offload_decision = 0
        self.number_of_eMBB_users = len(self.eMBB_Users)
        self.number_of_users = len(self.eMBB_Users) 
        self.number_of_urllc_users = len(self.URLLC_Users)
        self.num_allocate_RB_upper_bound = self.Communication_Channel_1.num_allocate_RBs_upper_bound
        self.num_allocate_RB_lower_bound = self.Communication_Channel_1.num_allocate_RBs_lower_bound
        self.time_divisions_per_slot = self.Communication_Channel_1.time_divisions_per_slot
        #self.max_transmit_power_db = 100#self.eMBB_UE_1.max_transmission_power_dBm
        self.max_transmit_power_db = 20
        self.min_transmit_power_db = 5
        self.offload_decisions_label = 0
        self.allocate_num_RB_label = 4
        self.allocate_transmit_powers_label = 1
        self.num_urllc_users_per_RB_label = 3
        self.total_energy = 0
        self.total_rate = 0
        self.selected_offload_decisions = []
        self.selected_powers = []
        self.selected_RBs = []
        self.powers = []
        self.subcarriers = []
        self.offload_decisions = []
        self.selected_actions = []
        self.rewards = []
        self.sum_allocations_per_RB_matrix = []
        self.RB_allocation_matrix = []
        self.resource_block_allocation_matrix = []
        self.resource_allocation_constraint_violation = 0

        #Define upper and lower bounds of observation and action spaces
        
        '''action_space_high = np.array([[self.max_offload_decision for _ in range(self.number_of_users)], [self.num_allocate_subcarriers_upper_bound for _ in range(self.number_of_users)], 
                        [self.max_transmit_power_db for _ in range(self.number_of_users)], [self.max_number_of_URLLC_users_per_RB for _ in range(self.number_of_users)]], dtype=np.float32)'''
        
        action_space_high = np.array([[1 for _ in range(self.number_of_users)], [1 for _ in range(self.number_of_users)]], dtype=np.float32)

        '''action_space_low = np.array([[self.min_offload_decision for _ in range(self.number_of_users)], [self.num_allocate_subcarriers_lower_bound for _ in range(self.number_of_users)], 
                        [self.min_transmit_power_db for _ in range(self.number_of_users)], [self.min_number_of_URLLC_users_per_RB for _ in range(self.number_of_users)]],dtype=np.float32)'''
        
        action_space_low = np.array([[0 for _ in range(self.number_of_users)], [0 for _ in range(self.number_of_users)]],dtype=np.float32)
        self.number_of_box_actions = 2
        action_space_high = np.transpose(action_space_high)
        action_space_low = np.transpose(action_space_low)
        
        '''observation_space_high = np.array([[self.channel_gain_max for _ in range(self.number_of_users)], [self.communication_queue_max for _ in range(self.number_of_users)], 
                        [self.energy_harvested_max for _ in range(self.number_of_users)], [self.latency_requirement_max for _ in range(self.number_of_users)], 
                        [self.reliability_requirement_max for _ in range(self.number_of_users)]],dtype=np.float32)'''
        
        number_of_batteries_per_user = 1
        number_of_lc_queues_per_user = 1
        numbers_of_off_queues_per_user = 1
        number_of_arriving_urllc_packets = 1
        number_of_states_per_embb_user = self.num_allocate_RB_upper_bound*2 + number_of_batteries_per_user + numbers_of_off_queues_per_user + number_of_lc_queues_per_user
        number_of_states_per_urllc_user = self.num_allocate_RB_upper_bound*2

        
        # observation_space_high = observation_space_high.reshape(1,self.number_of_eMBB_users*number_of_states_per_embb_user+self.number_of_urllc_users*number_of_states_per_urllc_user)

        
        # observation_space_low = np.array([[[self.channel_gain_min for _ in range(self.num_allocate_RB_upper_bound)] +
        #                                    [self.channel_gain_max for _ in range(self.num_allocate_RB_upper_bound)] + 
        #                                    [self.battery_energy_min for _ in range(1)] +
        #                                    [self.min_off_queue_length for _ in range(1)] +
        #                                    [self.max_lc_queue_length for _ in range(1)]]*self.number_of_users +
        #                                    [self.channel_gain_min for _ in range(self.num_allocate_RB_upper_bound)]*self.number_of_urllc_users],dtype=np.float32)
        
        # observation_space_high = observation_space_high.reshape(1,self.number_of_eMBB_users*number_of_states_per_embb_user+self.number_of_urllc_users*number_of_states_per_urllc_user)
        observation_space_high, observation_space_low = self.observation_space_collection()
        '''observation_space_low = np.array([[self.channel_gain_min for _ in range(self.number_of_users)], [self.communication_queue_min for _ in range(self.number_of_users)], 
                        [self.energy_harvested_min for _ in range(self.number_of_users)], [self.latency_requirement_min for _ in range(self.number_of_users)], 
                        [self.reliability_requirement_min for _ in range(self.number_of_users)]],dtype=np.float32)'''
        
        '''
        observation_space_high = [[channel_gain_max],[communication_queue_max],[energy_harvested_max],[latency_requirement_max],[reliability_requirement_max]]
        observation_space_low = [[channel_gain_min],[communication_queue_min],[energy_harvested_min],[latency_requirement_min],[reliability_requirement_min]]

        action_space_high = [[max_offload_decision],[num_allocate_subcarriers_upper_bound],[max_transmit_power_db],[max_number_of_URLLC_users_per_RB]]
        action_space_low = [[min_offload_decision],[num_allocate_subcarriers_lower_bound],[min_transmit_power_db],[min_number_of_URLLC_users_per_RB]]
        '''
        resource_allocation_action_space_low = np.array([0 for _ in range(self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)],dtype=np.float32)
        resource_allocation_action_space_high = np.array([1 for _ in range(self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)],dtype=np.float32)
        self.box_action_space = spaces.Box(low=action_space_low,high=action_space_high)
        self.number_of_box_actions = 2
        self.box_action_space_len = 0
        self.binary_action_space = spaces.MultiBinary(self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
        #self.binary_action_space = spaces.Box(low=resource_allocation_action_space_low,high=resource_allocation_action_space_high)
        self.binary_action_space_len = 0

        q_action_low = 0  # Lower bound for each dimension
        q_action_high = float('inf')  # Upper bound for each dimension
        self.q_action_space = spaces.Box(low=q_action_low,high=q_action_high)

        # Combine the action spaces into a dictionary
        #self.action_space = self.box_action_space
        
        self.action_space = spaces.Dict({
            'box_actions': self.box_action_space,
            'binary_actions': self.binary_action_space,
            'q_action': self.q_action_space
        })

        #self.action_space = spaces.Box(low=action_space_low,high=action_space_high)
        self.observation_space = spaces.Box(low=observation_space_low, high=observation_space_high)
        self.total_action_space = []
        
        sample_action = self.action_space.sample()
        sample_observation = self.observation_space.sample()
        reshaped_action_for_model_training, reshaped_action_for_model_training2 = self.reshape_action_space_dict(sample_action)
        #print('reshaped_action_for_model_training: ', reshaped_action_for_model_training)
        t = 0
        #reshaped_observation_for_model_training = self.reshape_observation_space_for_model(sample_observation)

        self.action_space_dim = len(reshaped_action_for_model_training)#self.box_action_space.shape[1] + (self.num_allocate_RB_upper_bound*self.time_divisions_per_slot)

        # print('sample_observation')
        # print(sample_observation)
        self.observation_space_dim = len(sample_observation)
        #print('self.observation_space_dim: ', self.observation_space_dim)
      
        self.action_space_high = 1
        self.action_space_low = 0

        self.STEP_LIMIT = 100
        self.sleep = 0
        self.steps = 0
        self.initial_RB_bandwidth = self.Communication_Channel_1.RB_bandwidth_Hz
        self.RB_bandwidth = self.initial_RB_bandwidth
        self.num_RBs_allocated = 0
        self.q_actions = 0
        self.resource_block_action_matrix = []

        self.max_small_scale_channel_gain = 0
        self.min_small_scale_channel_gain = 0

        self.max_battery_energy_level = 5
        self.min_battery_energy_level = 0

        self.max_large_scale_channel_gain = 0
        self.min_large_scale_channel_gain = 0

        self.max_local_queue_length = 0
        self.min_local_queue_length = 0

        self.max_offloading_queue_length = 0
        self.min_offloading_queue_length = 0

    def observation_space_collection(self):
        observation_space_high = []
        for x in range(0,self.num_allocate_RB_upper_bound*self.number_of_users):
            observation_space_high.append(self.channel_gain_max)

        for x in range(0,self.num_allocate_RB_upper_bound*self.number_of_users):
            observation_space_high.append(self.channel_gain_max)

        for x in range(0,self.number_of_users):
            observation_space_high.append(self.battery_energy_max)

        for x in range(0,self.number_of_users):
            observation_space_high.append(self.max_off_queue_length)

        for x in range(0,self.number_of_users):
            observation_space_high.append(self.max_lc_queue_length)

        for x in range(0,self.number_of_urllc_users*self.num_allocate_RB_upper_bound):
            observation_space_high.append(self.channel_gain_max)

        for x in range(0,self.number_of_urllc_users*self.num_allocate_RB_upper_bound):
            observation_space_high.append(self.channel_gain_max)


        observation_space_low = []
        for x in range(0,self.num_allocate_RB_upper_bound*self.number_of_users):
            observation_space_low.append(self.channel_gain_min)

        for x in range(0,self.num_allocate_RB_upper_bound*self.number_of_users):
            observation_space_low.append(self.channel_gain_min)

        for x in range(0,self.number_of_users):
            observation_space_low.append(self.battery_energy_min)

        for x in range(0,self.number_of_users):
            observation_space_low.append(self.min_off_queue_length)

        for x in range(0,self.number_of_users):
            observation_space_low.append(self.min_lc_queue_length)

        for x in range(0,self.number_of_urllc_users*self.num_allocate_RB_upper_bound):
            observation_space_low.append(self.channel_gain_min)

        for x in range(0,self.number_of_urllc_users*self.num_allocate_RB_upper_bound):
            observation_space_low.append(self.channel_gain_min)

        observation_space_high = np.array(observation_space_high)
        observation_space_low = np.array(observation_space_low)

        #print('observation_space_high.shape: ', observation_space_high.shape)

        return observation_space_high, observation_space_low

        
        
    def reshape_observation_space_for_model(self,observation_space,observation_channel_gains_urllc):
    
        observation_space = np.transpose(observation_space)
        # print('observation_space after transpose: ', observation_space.shape)
        observation_space = observation_space.reshape(1,len(observation_space)*len(observation_space[0]))
        # print('observation_space after reshape: ')
        # print(observation_space)
        #observation_space = observation_space.squeeze()
        #observation_space = observation_space.reshape(1,len(observation_space))
        # print('observation_space after squeeze')
        # print(observation_space.shape)

        observation_channel_gains_urllc = np.transpose(observation_channel_gains_urllc)
        observation_channel_gains_urllc = observation_channel_gains_urllc.reshape(1,len(observation_channel_gains_urllc)*len(observation_channel_gains_urllc[0]))
        #observation_channel_gains_urllc = np.transpose(observation_channel_gains_urllc)

        # print('observation_space.shape: ', observation_space.shape)
        #print('observation_channel_gains_urllc.shape: ', observation_channel_gains_urllc.shape)
    
        # print('observation_space')
        # print(observation_space)
        # print('shape: ', observation_space.shape)
        observation_space = np.column_stack((observation_space,observation_channel_gains_urllc))
        observation_space = observation_space.squeeze()
        # print('observation_space.shape: ', observation_space.shape)
        # print('')
        return observation_space
    
    def reshape_observation_space_for_model_normalizing(self,observation_space):
        observation_space = np.transpose(observation_space)
        observation_space = observation_space.reshape(1,len(observation_space)*len(observation_space[0]))
        observation_space = observation_space.squeeze()

        # print('observation_space')
        # print(observation_space)
        # print('shape: ', observation_space.shape)
        return observation_space
       
    def reshape_action_space_dict(self,action):
        box_action = np.array(action['box_actions'])
        binary_actions = np.array(action['binary_actions'])
        q_action = np.array(action['q_action'])

        len_box_actions = len(box_action) * len(box_action[0])
        self.box_action_space_len = len_box_actions

        box_action = box_action.reshape(1,len_box_actions)
        box_action = box_action.squeeze()

        binary_actions = binary_actions.reshape(1,self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
        binary_actions = binary_actions.squeeze()
        self.binary_action_space_len = self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound
        self.total_action_space = np.hstack((box_action,binary_actions))#np.column_stack((box_action,binary_actions))
        self.total_action_space = np.array(self.total_action_space)
        self.total_action_space = self.total_action_space.squeeze()
        self.total_action_space = np.append(self.total_action_space,q_action)

        action_space_dict = {
            'box_actions': box_action,
            'binary_actions': binary_actions,
            'q_action': q_action
        }
  
        return self.total_action_space, action_space_dict
    
    def reshape_action_space_for_model(self,action):
        box_action = np.array(action['box_actions'])
        binary_actions = np.array(action['binary_actions'])
        q_action = np.array(action['q_action'])

        #len_box_actions = len(box_action) * len(box_action[0])
        #self.box_action_space_len = len_box_actions

        #box_action = box_action.reshape(1,len_box_actions)
        #box_action = box_action.squeeze()

        #binary_actions = binary_actions.reshape(1,self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
        #binary_actions = binary_actions.squeeze()
        #self.binary_action_space_len = self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound
        self.total_action_space = np.hstack((box_action,binary_actions))#np.column_stack((box_action,binary_actions))
        self.total_action_space = np.array(self.total_action_space)
        self.total_action_space = self.total_action_space.squeeze()
        self.total_action_space = np.append(self.total_action_space,q_action)

  
        return self.total_action_space

        

    def reshape_action_space_from_model_to_dict(self,action):
        box_actions = []
        binary_actions = []
        q_action = []
        box_actions = action[0:self.box_action_space_len]
        binary_actions = action[self.box_action_space_len:len(action)-1]
        q_action.append(action[len(action)-1])

        box_actions = np.array(box_actions)
        binary_actions = np.array(binary_actions)
        q_action = np.array(q_action)

        #binary_actions = binary_actions.reshape(1,self.number_of_users * self.num_allocate_RB_upper_bound*self.time_divisions_per_slot).squeeze()
        #print(binary_actions)
 
        #print(binary_actions)
        action_space_dict = {
            'box_actions': box_actions,
            'binary_actions': binary_actions,
            'q_action': q_action
        }

        return action_space_dict

    def check_resource_block_allocation_constraint(self, binary_actions):
        resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.time_divisions_per_slot, self.num_allocate_RB_upper_bound)
        done_sampling = False
        resource_allocation_penalty = 0
        if not np.all(np.sum(np.sum(resource_block_action_matrix,axis=0),axis=0) <= self.time_divisions_per_slot):
            #resource_allocation_penalty = -0.02
            self.resource_allocation_constraint_violation+=1

        #return resource_allocation_penalty

    def enforce_constraint(self,action):
        box_actions = action['box_actions']
        binary_actions = action['binary_actions']
        q_action = action['q_action']
        resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.time_divisions_per_slot, self.num_allocate_RB_upper_bound)
        resource_block_action_matrix_size = self.number_of_users*self.time_divisions_per_slot*self.num_allocate_RB_upper_bound
        # main_column_array = resource_block_action_matrix[:,:,0]
        # #column_array = column_array.reshape(1,self.number_of_users*self.time_divisions_per_slot)
        # first_column_array = main_column_array[:,0]
        # second_column_array = main_column_array[:,1]
        # limit_index_array_1 = len(first_column_array)
        # limit_index_array_2 = len(second_column_array)
        # index_array_1 = list(range(0, limit_index_array_1))
        # index_array_2 = list(range(0, limit_index_array_2))
        # rand_num_1 = np.random.randint(0, len(index_array_1), 1)
        # rand_num_2 = np.random.randint(0, len(index_array_2), 1)
        # rand_num_1 = rand_num_1[0]
        # rand_num_2 = rand_num_2[0]
        # first_num = index_array_1[rand_num_1]
        # #index_array_1 = np.delete(index_array_1,rand_num,axis=0)
        # # rand_num = np.random.randint(0, len(index_array), 1)
        # # rand_num = rand_num[0]
        # second_num = index_array_2[rand_num_2]
        # index_first_num = first_num
        # index_second_num = second_num
        # main_column_array = [[0 for _ in range(self.time_divisions_per_slot)] for _ in range(limit_index_array_1)]
        # main_column_array[index_first_num][0] = 1
        # main_column_array[index_second_num][1] = 1
        # main_column_array = np.array(main_column_array)
        # resource_block_action_matrix[:,:,0] = main_column_array
        # print('resource_block_action_matrix')
        # print(resource_block_action_matrix)
        # #resource_block_action_matrix = resource_block_action_matrix.squeeze()
        # print(resource_block_action_matrix[:,:,0])
        # print('main_column_array')
        # print(main_column_array)
        # print('first_column_array')
        # print(first_column_array)
        # print('second_column_array')
        # print(second_column_array)
        # print('index_first_num: ', index_first_num)
        # print('index_second_num: ', index_second_num)
        # print('****************************************************')
        for z in range(0,self.num_allocate_RB_upper_bound):
            main_column_array = resource_block_action_matrix[:,:,z]
            #column_array = column_array.reshape(1,self.number_of_users*self.time_divisions_per_slot)
            first_column_array = main_column_array[:,0]
            second_column_array = main_column_array[:,1]
            limit_index_array_1 = len(first_column_array)
            limit_index_array_2 = len(second_column_array)
            index_array_1 = list(range(0, limit_index_array_1))
            index_array_2 = list(range(0, limit_index_array_2))
            rand_num_1 = np.random.randint(0, len(index_array_1), 1)
            rand_num_2 = np.random.randint(0, len(index_array_2), 1)
            rand_num_1 = rand_num_1[0]
            rand_num_2 = rand_num_2[0]
            first_num = index_array_1[rand_num_1]
            #index_array_1 = np.delete(index_array_1,rand_num,axis=0)
            # rand_num = np.random.randint(0, len(index_array), 1)
            # rand_num = rand_num[0]
            second_num = index_array_2[rand_num_2]
            index_first_num = first_num
            index_second_num = second_num
            main_column_array = [[0 for _ in range(self.time_divisions_per_slot)] for _ in range(limit_index_array_1)]
            main_column_array[index_first_num][0] = 1
            main_column_array[index_second_num][1] = 1
            main_column_array = np.array(main_column_array)
            resource_block_action_matrix[:,:,z] = main_column_array
            # count = 0
            # for x in range(0,self.number_of_users):
            #     for y in range(0,(self.time_divisions_per_slot)):
            #         if count == index_first_num or count == index_second_num:
            #             resource_block_action_matrix[x,y,z] = 1
            #         else:
            #             resource_block_action_matrix[x,y,z] = 0
            #         count+=1
        # print('****************************************************')
        # print('resource_block_action_matrix')
        # print(resource_block_action_matrix)
        resource_block_action_matrix = binary_actions.reshape(1, self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
        resource_block_action_matrix = resource_block_action_matrix.squeeze()
        #print(resource_block_action_matrix)
        action_space_dict = {
            'box_actions': box_actions,
            'binary_actions': resource_block_action_matrix,
            'q_action': q_action
        }
        #print(resource_block_action_matrix)
    
        return action
    
    def apply_resource_allocation_constraint(self,action,mode):
        box_actions = action['box_actions']
        binary_actions = action['binary_actions']
        q_action = action['q_action']
        #matrix = [[[random.uniform(0, 1) for _ in range(6)] for _ in range(2)] for _ in range(3)]
        #matrix = np.array(matrix)
        #print('matrix')
        #print(matrix)
        resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.time_divisions_per_slot, self.num_allocate_RB_upper_bound)
        resource_block_action_matrix_size = self.number_of_users*self.time_divisions_per_slot*self.num_allocate_RB_upper_bound
        #resource_block_action_matrix = resource_block_action_matrix.squeeze()
        #print(resource_block_action_matrix[:,:,0])
        # if mode == 'inference':
        #     print('resource_block_action_matrix before:')
        #     print(resource_block_action_matrix)
        for z in range(0,self.num_allocate_RB_upper_bound):
            main_column_array = resource_block_action_matrix[:,:,z]
            #print('main_column_array: ', main_column_array)
            first_column_array = main_column_array[:,0]
            second_column_array = main_column_array[:,1]
            #print('first_column_array: ', first_column_array)
            #print('second_column_array: ', second_column_array)
            limit_index_array_1 = len(first_column_array)
            #print('limit_index_array_1: ', limit_index_array_1)
            # column_array = column_array.reshape(1,self.number_of_users*self.time_divisions_per_slot)
            # column_array = column_array.squeeze()
            sorted_column_array_1 = np.sort(first_column_array)[::-1]
            sorted_column_array_2 = np.sort(second_column_array)[::-1]
            #print('sorted_column_array_1: ', sorted_column_array_1)
            #print('sorted_column_array_2: ', sorted_column_array_2)
            first_largest_num = sorted_column_array_1[0]
            second_largest_num = sorted_column_array_2[0]
            #print('first_largest_num: ', first_largest_num)
            #print('second_largest_num: ', second_largest_num)
            #print('np.where(first_column_array==first_largest_num): ', np.where(first_column_array==first_largest_num)[0])
            index_first_largest_num = np.where(first_column_array==first_largest_num)[0][0]
            #print('index_first_largest_num: ', index_first_largest_num)
            index_second_largest_num = np.where(second_column_array==second_largest_num)[0][0]
            #print('index_second_largest_num: ', index_second_largest_num)
            main_column_array = [[0 for _ in range(self.time_divisions_per_slot)] for _ in range(limit_index_array_1)]
            main_column_array[index_first_largest_num][0] = 1
            main_column_array[index_second_largest_num][1] = 1
            main_column_array = np.array(main_column_array)
            resource_block_action_matrix[:,:,z] = main_column_array
            count = 0
            # column_array = resource_block_action_matrix[:,:,z]
            # column_array = column_array.reshape(1,self.number_of_users*self.time_divisions_per_slot)
            # column_array = column_array.squeeze()
            # sorted_column_array = np.sort(column_array)[::-1]
            # first_largest_num = sorted_column_array[0]
            # second_largest_num = sorted_column_array[1]
            # index_first_largest_num = np.where(column_array==first_largest_num)[0][0]
            # index_second_largest_num = np.where(column_array==second_largest_num)[0][0]
            # count = 0
            # for x in range(0,self.number_of_users):
            #     for y in range(0,(self.time_divisions_per_slot)):
            #         if count == index_first_largest_num or count == index_second_largest_num:
            #             resource_block_action_matrix[x,y,z] = 1
            #         else:
            #             resource_block_action_matrix[x,y,z] = 0
            #         count+=1
        
        #if mode == 'inference':
            # print('resource_block_action_matrix after:')
            # print(resource_block_action_matrix)
            # print('')
        resource_block_action_matrix = binary_actions.reshape(1, self.number_of_users * self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
        resource_block_action_matrix = resource_block_action_matrix.squeeze()
        self.resource_block_action_matrix = resource_block_action_matrix

        action_space_dict = {
            'box_actions': box_actions,
            'binary_actions': resource_block_action_matrix,
            'q_action': q_action
        }
        #print(resource_block_action_matrix)
        return action_space_dict


    def user_binary_resource_allocations(self,user_resource_block_allocations):
        user_id = 0
        for eMBB_user in self.eMBB_Users:
            user_id = eMBB_user.eMBB_UE_label
    
    # def modify_resource_allocation_actions(self,binary_actions):
    #     binary_actions_mapped = []
    #     for binary_action in binary_actions:
    #         binary_actions_mapped.append(int(interp(binary_action,[0,1],[1,self.number_of_users])))
        
    #     binary_actions_mapped = np.array(binary_actions_mapped)
    #     resource_block_action_matrix = binary_actions_mapped.reshape(self.time_divisions_per_slot, self.num_allocate_RB_upper_bound)

    def step(self,action):
        #g = self.reshape_action_space_for_model(action)
        #print('action reshaped')
        #print(g)
        #.self.selected_actions =
        #print('------------')
        #f = self.reshape_action_space_for_model(action)
        #print(f)
        #  []
        #print(action)
        #self.reshape_action_space_for_model(action)
        #action = self.enforce_constraint(action)
        box_action = np.array(action['box_actions'])
        binary_actions = action['binary_actions']
        q_action = action['q_action']
        self.q_actions = q_action
        #print('q_action: ', q_action)
        #print('action')
        #print(action)
        #user_resource_block_allocations = action['user_resource_block_allocations']
        #user_resource_block_allocations = user_resource_block_allocations.reshape(self.time_divisions_per_slot,self.num_allocate_RB_upper_bound)
 

        self.check_resource_block_allocation_constraint(binary_actions)
    
        resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
    
        self.resource_block_allocation_matrix.append(resource_block_action_matrix)
        #print('resource_block_action_matrix')
        #print(resource_block_action_matrix)
        #print(' ')
        # done_sampling = True
        # if not np.all(np.sum(resource_block_action_matrix, axis=0) <= 1):
        #     while done_sampling:
        #         action = self.action_space.sample()
        #         box_action = np.array(action['box_actions'])
        #         binary_actions = action['binary_actions']
        #         resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.num_allocate_RB_upper_bound)
        #         if not np.all(np.sum(resource_block_action_matrix, axis=0) <= 1):
        #             done_sampling = True
        #         else:
        #             done_sampling = False

        #print(resource_block_action_matrix)
        #print(" ")
        #print("Action before interpolation")
        #print(action)
        #box_action = np.transpose(box_action)
        #print("Action before interpolation transposed")
        #print(action)
        reward = 0

        #collect offload decisions actions 
        num_offloading_actions = int(self.box_action_space_len/self.number_of_box_actions)

        num_power_action = num_offloading_actions

        # print('box_action')
        # print(box_action)
        offload_decisions_actions = box_action[0:num_offloading_actions]
        # print('offload_decisions_actions')
        # print(offload_decisions_actions)
     
        #offload_decisions_actions = offload_decisions_actions[0:self.number_of_eMBB_users]

        offload_decisions_actions_mapped = []
        for offload_decision in offload_decisions_actions:
            offload_decision_mapped = interp(offload_decision,[0,1],[self.min_offload_decision,self.max_offload_decision])
            offload_decisions_actions_mapped.append(offload_decision_mapped)
        
        #collect trasmit powers allocations actions
        transmit_power_actions = box_action[num_offloading_actions:num_offloading_actions*self.number_of_box_actions]
        #transmit_power_actions = transmit_power_actions[0:self.number_of_eMBB_users]

        transmit_power_actions_mapped = []

        for transmit_power_action in transmit_power_actions:
            transmit_power_action_mapped = interp(transmit_power_action,[0,1],[self.min_transmit_power_db,self.max_transmit_power_db])
            transmit_power_actions_mapped.append(transmit_power_action_mapped)

        #self.selected_powers.append(transmit_power_actions_mapped[0])
        
    
        #binary_actions = action['binary_actions']
        #resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.num_allocate_RB_upper_bound)
    
        RB_allocation_actions = resource_block_action_matrix 
        RB_sum_allocations = []
        for RB_allocation_action in RB_allocation_actions:
            RB_sum_allocations.append(sum(RB_allocation_action))


        #print(RB_allocation_actions)
        #RB_allocation_actions = RB_allocation_actions[0:self.number_of_eMBB_users]
        #RB_allocation_actions_mapped = []
        #print('RB_allocation_actions', RB_allocation_actions)
        #for RB_allocation_action in RB_allocation_actions:
        #    RB_allocation_action_mapped = interp(RB_allocation_action,[0,1],[self.num_allocate_RB_lower_bound,self.num_allocate_RB_upper_bound])
        #    RB_allocation_actions_mapped.append(RB_allocation_action_mapped)

        #RB_allocation_actions = (np.rint(RB_allocation_actions)).astype(int)
        #RB_allocation_actions_mapped = (np.rint(RB_allocation_actions_mapped)).astype(int)
        #print('RB_allocation_action_mapped: ', RB_allocation_actions_mapped)
        #self.selected_RBs.append(RB_allocation_actions_mapped[0]) 
        #self.selected_actions.append(RB_allocation_actions_mapped)
        
        #self.selected_actions.append(transmit_power_actions_mapped)
        
        #collect the final action - number of URLLC users per RB
        
        #print('Action after interpolation transposed')
        #offload_decisions_actions_mapped = [0,0,0,0,0,0,0,0,0,0,0]#[0, 0, 0.5, 0.5, 1, 1, 1]
        #transmit_power_actions_mapped = [20]#,20,20,20,20,20,20]
        #RB_allocation_actions = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])#, [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]])
        #print('RB_allocation_actions: ', RB_allocation_actions)
        #print(RB_allocation_actions)
        #RB_allocation_actions_mapped = [6]#,10,15,15,20,20,20]
        #number_URLLC_Users_per_RB_action_mapped = 3
        #print("New Timestep: ", self.steps)
        #print("offload_decisions_actions")
        #print(offload_decisions_actions_mapped)
        #print("subcarrier_allocation_actions")
        #print(subcarrier_allocation_actions_mapped)
        #print("transmit_power_actions")
        #print(transmit_power_actions_mapped)
        #print(' ')
        #print("number_URLLC_Users_per_RB_action")
        #print(number_URLLC_Users_per_RB_action_mapped)
        self.offload_decisions = offload_decisions_actions_mapped
        self.powers = transmit_power_actions_mapped
        self.subcarriers = RB_sum_allocations
        self.RB_allocation_matrix = RB_allocation_actions

        #print('self.offload decisions')
        #print(offload_decision_mapped)
        #available_resource_time_code_block_fn
        #Perform Actions
        self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions_mapped)
        #self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions)
        #offload_decisions_actions_mapped = [1]
        #offload_decisions_actions_mapped = np.array(offload_decisions_actions_mapped)
        #print('offload_decisions_actions_mapped: ', offload_decisions_actions_mapped)
        self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions_mapped)
        #self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions)

        #self.Communication_Channel_1.number_URLLC_Users_per_RB = number_URLLC_Users_per_RB_action_mapped
        #self.Communication_Channel_1.number_URLLC_Users_per_RB = number_URLLC_Users_per_RB_action

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS1)
        self.Communication_Channel_1.initiate_RBs()
        #RB_allocation_actions = [[1,0,0,0,0,0,0,0,0,0,0,0]]
        # RB_allocation_actions = np.array(RB_allocation_actions)
        # print('RB_allocation_actions')
        # print(RB_allocation_actions)
        # print('')
        self.num_RBs_allocated = sum(RB_allocation_actions[0])
        #print("RB_allocation_actions: ", RB_allocation_actions)
        self.Communication_Channel_1.allocate_RBs_eMBB(self.eMBB_Users,RB_allocation_actions)
        #self.Communication_Channel_1.allocate_subcarriers_eMBB(self.eMBB_Users,subcarrier_allocation_actions)
        #self.Communication_Channel_1.create_resource_blocks_URLLC()
        #self.Communication_Channel_1.allocate_resource_blocks_URLLC(self.URLLC_Users)
        #self.Communication_Channel_1.subcarrier_URLLC_User_mapping()


        for eMBB_User in self.eMBB_Users:
            eMBB_User.increment_task_queue_timers()
            eMBB_User.split_tasks()
            eMBB_User.available_resource_time_code_block_fn(self.Communication_Channel_1)
       
        self.SBS1.allocate_resource_blocks_URLLC(self.Communication_Channel_1, self.URLLC_Users, self.eMBB_Users)
        # for urllc_user in self.URLLC_Users:
        #     print('urllc_user.assigned_resource_block: ', urllc_user.assigned_resource_block, 'urllc_user.assigned_time_block: ', urllc_user.assigned_time_block,
        #           'urllc_user.assigned_code_block: ', urllc_user.assigned_code_block)
                        

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
        for eMBB_User in self.eMBB_Users: 
            eMBB_User.urllc_puncturing_users_sum_data_rates(self.URLLC_Users)

        self.SBS1.receive_offload_packets(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_energy_consumption(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_processing_delay(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_rate_eMBB_users(self.eMBB_Users)
        self.SBS1.calculate_achieved_system_energy_efficiency()
        system_reward, reward, self.total_energy,self.total_rate = self.SBS1.calculate_achieved_system_reward(self.eMBB_Users,self.URLLC_Users,self.Communication_Channel_1, q_action)
    
        #reward = [x + resource_block_allocation_penalty for x in reward]
        # print('self.max_battery_energy_level')
        # print(self.max_battery_energy_level)
        #print('Reward')
        #print(reward)
        #print(' ')
        #mapped_reward = interp(reward,[0,1000],[7200000000,7830000000])
        #Update game state after performing actions
        for eMBB_User in self.eMBB_Users:
            #eMBB_User.calculate_distance_from_SBS(self.SBS1.x_position, self.SBS1.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
            eMBB_User.calculate_channel_gain(self.Communication_Channel_1)
            eMBB_User.harvest_energy()
            eMBB_User.compute_battery_energy_level()
            eMBB_User.generate_task(self.Communication_Channel_1)
            eMBB_User.collect_state()

        for urllc_user in self.URLLC_Users:
            urllc_user.calculate_channel_gain_on_all_resource_blocks(self.Communication_Channel_1)
            urllc_user.generate_task(self.Communication_Channel_1)
            urllc_user.split_tasks()
            urllc_user.collect_state()

        observation_channel_gains, observation_battery_energies, observation_offloading_queue_lengths, observation_local_queue_lengths, num_urllc_arriving_packets, observation_channel_gains_urllc = self.SBS1.collect_state_space(self.eMBB_Users, self.URLLC_Users)
        
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
        observation_channel_gains = np.array(observation_channel_gains)
        # print('observation_channel_gains:')
        # print(observation_channel_gains)
        # print('observation_channel_gains.shape: ', observation_channel_gains.shape)
        user = 0
        for channel_gains in observation_channel_gains:
            #print('channel_gains: ', channel_gains)
            for channel_gain in channel_gains:
                # print('channel_gain')
                # print(channel_gain)
                for gain in channel_gain:
                    if col < self.num_allocate_RB_upper_bound:
                        observation_channel_gains[user][0][col] = interp(observation_channel_gains[user][0][col],[self.min_small_scale_channel_gain,self.max_small_scale_channel_gain],[0,1])
                    else:
                        observation_channel_gains[user][0][col] = interp(observation_channel_gains[user][0][col],[self.min_large_scale_channel_gain,self.max_large_scale_channel_gain],[0,1])
                    col+=1

                row+=1
                col = 0
            user+=1
  
        row = 0

        # print('observation_channel_gains.shape: ', observation_channel_gains.shape)
        observation_channel_gains_urllc = np.array(observation_channel_gains_urllc)
        # print('observation_channel_gains_urllc')
        # print(observation_channel_gains_urllc)
        user = 0
        for channel_gains in observation_channel_gains_urllc:
            #print('channel_gains: ', channel_gains)
            for channel_gain in channel_gains:
                # print('channel_gain')
                # print(channel_gain)
                for gain in channel_gain:
                    if col < self.num_allocate_RB_upper_bound:
                        observation_channel_gains_urllc[user][0][col] = interp(observation_channel_gains_urllc[user][0][col],[self.min_small_scale_channel_gain,self.max_small_scale_channel_gain],[0,1])
                    else:
                        observation_channel_gains_urllc[user][0][col] = interp(observation_channel_gains_urllc[user][0][col],[self.min_large_scale_channel_gain,self.max_large_scale_channel_gain],[0,1])
                    col+=1

                row+=1
                col = 0
            user+=1
  
        row = 0
        # print('observation_channel_gains_urllc: ')
        # print(observation_channel_gains_urllc)
        for battery_energy in observation_battery_energies:
            observation_battery_energies[row] = interp(observation_battery_energies[row],[self.min_battery_energy_level,self.max_battery_energy_level],[0,1])
            row+=1
        
        row = 0
        for offloading_queue_length in observation_offloading_queue_lengths:
            # print('offload_queue_length: ', observation_offloading_queue_lengths[row])
            # print('self.min_offloading_queue_length: ', self.min_offloading_queue_length)
            # print('self.max_offloading_queue_length: ', self.max_offloading_queue_length)
            observation_offloading_queue_lengths[row] = interp(observation_offloading_queue_lengths[row],[self.min_offloading_queue_length,self.max_offloading_queue_length],[0,1])
            row+=1

        row = 0
        for local_queue_length in observation_local_queue_lengths:
            observation_local_queue_lengths[row] = interp(observation_local_queue_lengths[row],[self.min_local_queue_length,self.max_local_queue_length],[0,1])
            row+=1

        observation_channel_gains = np.array(observation_channel_gains).squeeze()
        observation_channel_gains_urllc = np.array(observation_channel_gains_urllc).squeeze()
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
        # print('observation_battery_energies:')
        # print(observation_channel_gains_urllc.shape)
        #if len(observation_channel_gains_urllc) > 0:
        #    observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths,observation_channel_gains_urllc)) #observation_channel_gains.
        #else:
        observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths)) #observation_channel_gains.
    
        #print('observation matrix')
        #print('observation.shape: ', observation.shape)
        observation = self.reshape_observation_space_for_model(observation,observation_channel_gains_urllc)
        # print('observation: ')
        # print(observation)
       

        done = self.check_timestep()
        dones = [0 for element in range(len(self.eMBB_Users) - 1)]
        dones.append(done)
        info = {'reward': reward}
        self.steps+=1
        #print('Timestep: ', self.steps)
        #print('reward: ', reward)
        self.rewards.append(reward)
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
        self.timestep_counter+=1
        self.RB_bandwidth = self.Communication_Channel_1.RB_bandwidth_Hz
        return observation,reward,done,info
    
    def reset(self):
        self.steps = 0
        self.SBS1.set_properties()
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
        self.battery_energy_max = self.eMBB_UE_1.max_battery_capacity
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
        # if self.timestep_counter >= 500000:
        #     self.Communication_Channel_1.RB_bandwidth_Hz = 10*self.initial_RB_bandwidth
       
        for eMBB_User in self.eMBB_Users:
            #eMBB_User.set_properties_UE()
            eMBB_User.set_properties_eMBB()
            eMBB_User.collect_state()
            eMBB_User.calculate_distance_from_SBS(self.SBS1.x_coordinate,self.SBS1.y_coordinate)

        for URLLC_User in self.URLLC_Users:
            URLLC_User.set_properties_UE()
            URLLC_User.set_properties_URLLC()
            URLLC_User.calculate_distances_from_embb_users(self.eMBB_Users)
            URLLC_User.calculate_distance_from_SBS(self.SBS1.x_coordinate, self.SBS1.y_coordinate)
            URLLC_User.collect_state()

        self.eMBB_Users.clear()
        self.URLLC_Users.clear()
        self.group_users()

        self.SBS1.associate_users(self.eMBB_Users, self.URLLC_Users)
        self.Communication_Channel_1.set_properties()

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS1)
        self.Communication_Channel_1.initiate_RBs()
        self.SBS1.allocate_resource_blocks_URLLC(self.Communication_Channel_1, self.URLLC_Users,self.eMBB_Users)
        
        info = {'reward': 0}
        #print('battery enegy: ', self.SBS1.system_state_space[4])
        #observation_channel_gains, observation_battery_energies = self.SBS1.collect_state_space(self.eMBB_Users)
        observation_channel_gains, observation_battery_energies, observation_offloading_queue_lengths, observation_local_queue_lengths, num_urllc_arriving_packets, observation_channel_gains_urllc = self.SBS1.collect_state_space(self.eMBB_Users, self.URLLC_Users)
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
        # print('observation_channel_gains')
        # print(observation_channel_gains)
        for channel_gains in observation_channel_gains:
            for channel_gain in channel_gains:
                observation_channel_gains[row][col] = interp(observation_channel_gains[row][col],[self.channel_gain_min,self.channel_gain_max],[0,1])
                col+=1

            row+=1
            col = 0
        row = 0
        # print('observation_channel_gains_urllc')
        # print(observation_channel_gains_urllc)
        for channel_gains in observation_channel_gains_urllc:
            for channel_gain in channel_gains:
                observation_channel_gains_urllc[row][col] = interp(observation_channel_gains_urllc[row][col],[self.channel_gain_min,self.channel_gain_max],[0,1])
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
        #if len(observation_channel_gains_urllc) > 0:
        #    observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths,observation_channel_gains_urllc)) #observation_channel_gains.
        #else:
        observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths)) #observation_channel_gains.
        observation = self.reshape_observation_space_for_model(observation,observation_channel_gains_urllc)
        reward = 0
        done = 0
        return observation

    def render(self, mode='human'):
        pass

    def create_objects(self):
        #Small Cell Base station
        self.SBS1 = SBS(1)
        self.eMBB_Users = []
        self.URLLC_Users = []
  
        #Users
        self.eMBB_UE_1 = eMBB_UE(1,1,100,600)
        self.eMBB_UE_2 = eMBB_UE(2,2,100,600)
        self.eMBB_UE_3 = eMBB_UE(3,3,100,600)
        self.eMBB_UE_4 = eMBB_UE(4,4,100,600)
        self.eMBB_UE_5 = eMBB_UE(5,5,100,600)
        self.eMBB_UE_6 = eMBB_UE(6,6,100,600)
        self.eMBB_UE_7 = eMBB_UE(7,7,100,600)
        self.eMBB_UE_8 = eMBB_UE(8,8,100,600)
        self.eMBB_UE_9 = eMBB_UE(9,9,100,600)
        self.eMBB_UE_10 = eMBB_UE(10,10,100,600)
        self.eMBB_UE_11 = eMBB_UE(11,11,100,600)

        self.URLLC_UE_1 = URLLC_UE(1,12,100,600)
        self.URLLC_UE_2 = URLLC_UE(2,13,100,600)
        self.URLLC_UE_3 = URLLC_UE(3,14,100,600)
        self.URLLC_UE_4 = URLLC_UE(4,15,100,600)
        self.URLLC_UE_5 = URLLC_UE(5,16,100,600)
        self.URLLC_UE_6 = URLLC_UE(6,17,100,600)
        self.URLLC_UE_7 = URLLC_UE(7,18,100,600)
        self.URLLC_UE_8 = URLLC_UE(8,19,100,600)
        self.URLLC_UE_9 = URLLC_UE(9,20,100,600)
        self.URLLC_UE_10 = URLLC_UE(10,21,100,600)
        self.URLLC_UE_11 = URLLC_UE(11,22,100,600)
        self.URLLC_UE_12 = URLLC_UE(12,23,100,600)

        # Set channel gain scaling factorsS
        #self.eMBB_UE_4.set_channel_gain_scaling_factor(40)
        #self.eMBB_UE_4.set_channel_gain_scaling_factor(20)


        #Communication Channel
        self.Communication_Channel_1 = Communication_Channel(self.SBS1.SBS_label)

        #Group Users

        self.group_users()

        #Associate SBS with users
        self.SBS1.associate_users(self.eMBB_Users,self.URLLC_Users)

    def acquire_users(self,embb_users,urllc_users):
        self.eMBB_Users = embb_users
        self.URLLC_users = urllc_users 

    def group_users(self):
        #Group all eMBB Users
        self.eMBB_Users.append(self.eMBB_UE_1)
        self.eMBB_Users.append(self.eMBB_UE_2)
        self.eMBB_Users.append(self.eMBB_UE_3)
        self.eMBB_Users.append(self.eMBB_UE_4)
        self.eMBB_Users.append(self.eMBB_UE_5)
        self.eMBB_Users.append(self.eMBB_UE_6)
        #self.eMBB_Users.append(self.eMBB_UE_7)
        #self.eMBB_Users.append(self.eMBB_UE_8)
        #self.eMBB_Users.append(self.eMBB_UE_9)
        #self.eMBB_Users.append(self.eMBB_UE_10)
        #self.eMBB_Users.append(self.eMBB_UE_11)

        self.URLLC_Users.append(self.URLLC_UE_1)
        self.URLLC_Users.append(self.URLLC_UE_2)
        self.URLLC_Users.append(self.URLLC_UE_3)
        self.URLLC_Users.append(self.URLLC_UE_4)
        self.URLLC_Users.append(self.URLLC_UE_5)
        self.URLLC_Users.append(self.URLLC_UE_6)
        self.URLLC_Users.append(self.URLLC_UE_7)
        #self.URLLC_Users.append(self.URLLC_UE_8)
        #self.URLLC_Users.append(self.URLLC_UE_9)
        #self.URLLC_Users.append(self.URLLC_UE_10)
        #self.URLLC_Users.append(self.URLLC_UE_11)
        #self.URLLC_Users.append(self.URLLC_UE_12)


    def check_timestep(self):
        if self.steps >= self.STEP_LIMIT:
            return True
        else: 
            return False
        
    def seed(self):
        pass

    def step_(self, action):
        #g = self.reshape_action_space_for_model(action)
        #print('action reshaped')
        #print(g)
        #.self.selected_actions =
        #print('------------')
        #f = self.reshape_action_space_for_model(action)
        #print(f)
        #  []
        #print(action)
        #self.reshape_action_space_for_model(action)
        #action = self.enforce_constraint(action)
        box_action = np.array(action['box_actions'])
        binary_actions = action['binary_actions']
        q_action = action['q_action']
        self.q_actions = q_action
        #print('q_action: ', q_action)
        #print('action')
        #print(action)
        #user_resource_block_allocations = action['user_resource_block_allocations']
        #user_resource_block_allocations = user_resource_block_allocations.reshape(self.time_divisions_per_slot,self.num_allocate_RB_upper_bound)
 

        self.check_resource_block_allocation_constraint(binary_actions)
    
        resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.time_divisions_per_slot * self.num_allocate_RB_upper_bound)
    
        self.resource_block_allocation_matrix.append(resource_block_action_matrix)
        #print('resource_block_action_matrix')
        #print(resource_block_action_matrix)
        #print(' ')
        # done_sampling = True
        # if not np.all(np.sum(resource_block_action_matrix, axis=0) <= 1):
        #     while done_sampling:
        #         action = self.action_space.sample()
        #         box_action = np.array(action['box_actions'])
        #         binary_actions = action['binary_actions']
        #         resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.num_allocate_RB_upper_bound)
        #         if not np.all(np.sum(resource_block_action_matrix, axis=0) <= 1):
        #             done_sampling = True
        #         else:
        #             done_sampling = False

        #print(resource_block_action_matrix)
        #print(" ")
        #print("Action before interpolation")
        #print(action)
        #box_action = np.transpose(box_action)
        #print("Action before interpolation transposed")
        #print(action)
        reward = 0

        #collect offload decisions actions 
        num_offloading_actions = int(self.box_action_space_len/self.number_of_box_actions)

        num_power_action = num_offloading_actions

        # print('box_action')
        # print(box_action)
        offload_decisions_actions = box_action[0:num_offloading_actions]
        # print('offload_decisions_actions')
        # print(offload_decisions_actions)
     
        #offload_decisions_actions = offload_decisions_actions[0:self.number_of_eMBB_users]

        offload_decisions_actions_mapped = []
        for offload_decision in offload_decisions_actions:
            offload_decision_mapped = interp(offload_decision,[0,1],[self.min_offload_decision,self.max_offload_decision])
            offload_decisions_actions_mapped.append(offload_decision_mapped)
        
        #collect trasmit powers allocations actions
        transmit_power_actions = box_action[num_offloading_actions:num_offloading_actions*self.number_of_box_actions]
        #transmit_power_actions = transmit_power_actions[0:self.number_of_eMBB_users]

        transmit_power_actions_mapped = []

        for transmit_power_action in transmit_power_actions:
            transmit_power_action_mapped = interp(transmit_power_action,[0,1],[self.min_transmit_power_db,self.max_transmit_power_db])
            transmit_power_actions_mapped.append(transmit_power_action_mapped)

        #self.selected_powers.append(transmit_power_actions_mapped[0])
        
    
        #binary_actions = action['binary_actions']
        #resource_block_action_matrix = binary_actions.reshape(self.number_of_users, self.num_allocate_RB_upper_bound)
    
        RB_allocation_actions = resource_block_action_matrix 
        RB_sum_allocations = []
        for RB_allocation_action in RB_allocation_actions:
            RB_sum_allocations.append(sum(RB_allocation_action))


        #print(RB_allocation_actions)
        #RB_allocation_actions = RB_allocation_actions[0:self.number_of_eMBB_users]
        #RB_allocation_actions_mapped = []
        #print('RB_allocation_actions', RB_allocation_actions)
        #for RB_allocation_action in RB_allocation_actions:
        #    RB_allocation_action_mapped = interp(RB_allocation_action,[0,1],[self.num_allocate_RB_lower_bound,self.num_allocate_RB_upper_bound])
        #    RB_allocation_actions_mapped.append(RB_allocation_action_mapped)

        #RB_allocation_actions = (np.rint(RB_allocation_actions)).astype(int)
        #RB_allocation_actions_mapped = (np.rint(RB_allocation_actions_mapped)).astype(int)
        #print('RB_allocation_action_mapped: ', RB_allocation_actions_mapped)
        #self.selected_RBs.append(RB_allocation_actions_mapped[0]) 
        #self.selected_actions.append(RB_allocation_actions_mapped)
        
        #self.selected_actions.append(transmit_power_actions_mapped)
        
        #collect the final action - number of URLLC users per RB
        
        #print('Action after interpolation transposed')
        #offload_decisions_actions_mapped = [1,1,1,1,1,1,1,1,1]#[0, 0, 0.5, 0.5, 1, 1, 1]
        #transmit_power_actions_mapped = [20]#,20,20,20,20,20,20]
        #RB_allocation_actions = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])#, [0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]])
        #print('RB_allocation_actions: ', RB_allocation_actions)
        #print(RB_allocation_actions)
        #RB_allocation_actions_mapped = [6]#,10,15,15,20,20,20]
        #number_URLLC_Users_per_RB_action_mapped = 3
        #print("New Timestep: ", self.steps)
        #print("offload_decisions_actions")
        #print(offload_decisions_actions_mapped)
        #print("subcarrier_allocation_actions")
        #print(subcarrier_allocation_actions_mapped)
        #print("transmit_power_actions")
        #print(transmit_power_actions_mapped)
        #print(' ')
        #print("number_URLLC_Users_per_RB_action")
        #print(number_URLLC_Users_per_RB_action_mapped)
        self.offload_decisions = offload_decisions_actions_mapped
        self.powers = transmit_power_actions_mapped
        self.subcarriers = RB_sum_allocations
        self.RB_allocation_matrix = RB_allocation_actions

        #print('self.offload decisions')
        #print(offload_decision_mapped)

        #Perform Actions
        self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions_mapped)
        #self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions)
        #offload_decisions_actions_mapped = [1]
        #offload_decisions_actions_mapped = np.array(offload_decisions_actions_mapped)
        #print('offload_decisions_actions_mapped: ', offload_decisions_actions_mapped)
        self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions_mapped)
        #self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions)

        #self.Communication_Channel_1.number_URLLC_Users_per_RB = number_URLLC_Users_per_RB_action_mapped
        #self.Communication_Channel_1.number_URLLC_Users_per_RB = number_URLLC_Users_per_RB_action

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS1)
        self.Communication_Channel_1.initiate_RBs()
        #RB_allocation_actions = [[1,0,0,0,0,0,0,0,0,0,0,0]]
        # RB_allocation_actions = np.array(RB_allocation_actions)
        # print('RB_allocation_actions')
        # print(RB_allocation_actions)
        # print('')
        self.num_RBs_allocated = sum(RB_allocation_actions[0])
        #print("RB_allocation_actions: ", RB_allocation_actions)
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
        for eMBB_User in self.eMBB_Users: 
            eMBB_User.urllc_puncturing_users_sum_data_rates(self.URLLC_Users)

        self.SBS1.receive_offload_packets(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_energy_consumption(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_processing_delay(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_rate_eMBB_users(self.eMBB_Users)
        self.SBS1.calculate_achieved_system_energy_efficiency()
        system_reward, reward, self.total_energy,self.total_rate = self.SBS1.calculate_achieved_system_reward(self.eMBB_Users,self.URLLC_Users,self.Communication_Channel_1, q_action)
    
        #reward = [x + resource_block_allocation_penalty for x in reward]
       
        
        #print('Reward')
        #print(reward)
        #print(' ')
        #mapped_reward = interp(reward,[0,1000],[7200000000,7830000000])
        #Update game state after performing actions
        for eMBB_User in self.eMBB_Users:
            #eMBB_User.calculate_distance_from_SBS(self.SBS1.x_position, self.SBS1.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
            eMBB_User.calculate_channel_gain(self.Communication_Channel_1)
            eMBB_User.harvest_energy()
            eMBB_User.compute_battery_energy_level()
            eMBB_User.generate_task(self.Communication_Channel_1)
            eMBB_User.collect_state()

        for urllc_user in self.URLLC_Users:
            urllc_user.calculate_channel_gain_on_all_resource_blocks(self.Communication_Channel_1)
            urllc_user.generate_task(self.Communication_Channel_1)
            urllc_user.split_tasks()

        observation_channel_gains, observation_battery_energies, observation_offloading_queue_lengths, observation_local_queue_lengths, num_urllc_arriving_packets, observation_channel_gains_urllc = self.SBS1.collect_state_space(self.eMBB_Users, self.URLLC_Users)
        
        #observation_channel_gains, observation_battery_energies = self.SBS1.collect_state_space(self.eMBB_Users)
        #observation_channel_gains = np.array(observation_channel_gains, dtype=np.float32)
        #observation_battery_energies = np.array(observation_battery_energies, dtype=np.float32)
        #print('Observation before transpose')
        #print(np.transpose(observation))
        #normalize observation values to a range between 0 and 1 using interpolation
        # row = 0
        # for num_urllc_arriving_packet in num_urllc_arriving_packets:
        #     num_urllc_arriving_packets[row] = interp(num_urllc_arriving_packets[row],[0,len(self.URLLC_Users)],[0,1])
        #     row+=1
        # row = 0
        # col = 0
        # min_value = 0
        # max_value = 0
        # for channel_gains in observation_channel_gains:
        #     for channel_gain in channel_gains:
        #         observation_channel_gains[row][col] = interp(observation_channel_gains[row][col],[self.channel_gain_min,self.channel_gain_max],[0,1])
        #         col+=1

        #     row+=1
        #     col = 0
  
        # row = 0
        # for battery_energy in observation_battery_energies:
        #     observation_battery_energies[row] = interp(observation_battery_energies[row],[self.battery_energy_min,self.battery_energy_max],[0,1])
        #     row+=1
        
        # row = 0
        # for offloading_queue_length in observation_offloading_queue_lengths:
        #     observation_offloading_queue_lengths[row] = interp(observation_offloading_queue_lengths[row],[self.min_off_queue_length,self.max_off_queue_length],[0,1])
        #     row+=1

        # row = 0
        # for local_queue_length in observation_local_queue_lengths:
        #     observation_local_queue_lengths[row] = interp(observation_local_queue_lengths[row],[self.min_lc_queue_length,self.max_lc_queue_length],[0,1])
        #     row+=1

        observation_channel_gains = np.array(observation_channel_gains).squeeze()
        
        observation_battery_energies = np.array(observation_battery_energies)
        observation_offloading_queue_lengths = np.array(observation_offloading_queue_lengths)
        observation_local_queue_lengths = np.array(observation_local_queue_lengths)
        num_urllc_arriving_packets = np.array(num_urllc_arriving_packets)
        observation_channel_gains_urllc = np.array(observation_channel_gains_urllc).squeeze()

        if self.number_of_users == 1:
            observation_channel_gains_num = len(observation_channel_gains)
            observation_battery_energies_num = len(observation_battery_energies)
            #observation_offloading_queue_lengths_num = len(observation_offloading_queue_lengths)
            #observation_local_queue_lengths_num = len(observation_local_queue_lengths)

            observation_channel_gains = observation_channel_gains.reshape(observation_battery_energies_num,observation_channel_gains_num)
        
      
        #observation_channel_gains = np.transpose(observation_channel_gains)
        #observation_battery_energies = np.transpose(observation_battery_energies)
        #if len(observation_channel_gains_urllc) > 0:
        #observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths,observation_channel_gains_urllc)) #observation_channel_gains.
        #else:
        observation = np.column_stack((observation_channel_gains,observation_battery_energies,observation_offloading_queue_lengths,observation_local_queue_lengths)) #observation_channel_gains.
        #print('observation matrix')
        observation = self.reshape_observation_space_for_model_normalizing(observation)
        # print('observation: ')
        # print(observation)
       

        done = self.check_timestep()
        dones = [0 for element in range(len(self.eMBB_Users) - 1)]
        dones.append(done)
        info = {'reward': reward}
        self.steps+=1
        #print('Timestep: ', self.steps)
        #print('reward: ', reward)
        self.rewards.append(reward)
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
        self.timestep_counter+=1
        self.RB_bandwidth = self.Communication_Channel_1.RB_bandwidth_Hz
        return observation,reward,done,info
    
    def change_state_limits(self, min_small_scale_channel_gain,max_small_scale_channel_gain,
                            min_large_scale_channel_gain,max_large_scale_channel_gain,
                            min_battery_energy_level,max_battery_energy_level,
                            min_local_queueing_length,max_local_queueing_length,
                            min_offloading_queueing_length,max_offloading_queueing_length):
        self.min_small_scale_channel_gain = min_small_scale_channel_gain
        self.max_small_scale_channel_gain = max_small_scale_channel_gain
        self.min_large_scale_channel_gain = min_large_scale_channel_gain
        self.max_large_scale_channel_gain = max_large_scale_channel_gain
        self.max_battery_energy_level = max_battery_energy_level
        self.min_battery_energy_level = min_battery_energy_level
        self.min_local_queue_length = min_local_queueing_length
        self.max_local_queue_length = max_local_queueing_length
        self.min_offloading_queue_length = min_offloading_queueing_length
        self.max_offloading_queue_length = max_offloading_queueing_length