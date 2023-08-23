import gym
from gym import spaces
import pygame, sys, time, random, numpy as np
from eMBB_UE_2 import eMBB_UE
from Communication_channel_2 import Communication_Channel
from SBS import SBS
from numpy import interp

pygame.init()

#Set constant variables
ENV_WIDTH = 400 #400m
ENV_HEIGHT = 400 #400m

clock = pygame.time.Clock()
#screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

class NetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.create_objects()
        self.reset()
        #Action Space Bound Paramaters
        self.max_offload_decision = 1
        self.min_offload_decision = 0
        self.number_of_eMBB_users = len(self.eMBB_Users)
        self.number_of_users = len(self.eMBB_Users) 
        self.num_allocate_RB_upper_bound = self.Communication_Channel_1.num_allocate_RB_upper_bound
        self.num_allocate_RB_lower_bound = self.Communication_Channel_1.num_allocate_RB_lower_bound 
        self.max_transmit_power_db = self.eMBB_Users[0].max_transmission_power_dBm
        self.min_transmit_power_db = 0
        self.offload_decisions_label = 0
        self.allocate_num_RB_label = 1
        self.allocate_transmit_powers_label = 2

        #Define upper and lower bounds of observation and action spaces
        
        '''action_space_high = np.array([[self.max_offload_decision for _ in range(self.number_of_users)], [self.num_allocate_subcarriers_upper_bound for _ in range(self.number_of_users)], 
                        [self.max_transmit_power_db for _ in range(self.number_of_users)], [self.max_number_of_URLLC_users_per_RB for _ in range(self.number_of_users)]], dtype=np.float32)'''
        
        action_space_high = np.array([[1 for _ in range(self.number_of_users)], [1 for _ in range(self.number_of_users)], 
                        [1 for _ in range(self.number_of_users)]], dtype=np.float32)

        '''action_space_low = np.array([[self.min_offload_decision for _ in range(self.number_of_users)], [self.num_allocate_subcarriers_lower_bound for _ in range(self.number_of_users)], 
                        [self.min_transmit_power_db for _ in range(self.number_of_users)], [self.min_number_of_URLLC_users_per_RB for _ in range(self.number_of_users)]],dtype=np.float32)'''
        
        action_space_low = np.array([[0 for _ in range(self.number_of_users)], [0 for _ in range(self.number_of_users)], 
                        [0 for _ in range(self.number_of_users)]],dtype=np.float32)
        
        action_space_high = np.transpose(action_space_high)
        action_space_low = np.transpose(action_space_low)
        
        observation_space_high = np.array([[self.channel_gain_max for _ in range(self.number_of_users)], [self.communication_queue_max for _ in range(self.number_of_users)], 
                        [self.energy_harvested_max for _ in range(self.number_of_users)], [self.latency_requirement_max for _ in range(self.number_of_users)], 
                        [self.reliability_requirement_max for _ in range(self.number_of_users)]],dtype=np.float32)
        
        observation_space_low = np.array([[self.channel_gain_min for _ in range(self.number_of_users)], [self.communication_queue_min for _ in range(self.number_of_users)], 
                        [self.energy_harvested_min for _ in range(self.number_of_users)], [self.latency_requirement_min for _ in range(self.number_of_users)], 
                        [self.reliability_requirement_min for _ in range(self.number_of_users)]],dtype=np.float32)
        
        observation_space_high = np.transpose(observation_space_high)
        observation_space_low = np.transpose(observation_space_low)
        
        '''
        observation_space_high = [[channel_gain_max],[communication_queue_max],[energy_harvested_max],[latency_requirement_max],[reliability_requirement_max]]
        observation_space_low = [[channel_gain_min],[communication_queue_min],[energy_harvested_min],[latency_requirement_min],[reliability_requirement_min]]

        action_space_high = [[max_offload_decision],[num_allocate_subcarriers_upper_bound],[max_transmit_power_db],[max_number_of_URLLC_users_per_RB]]
        action_space_low = [[min_offload_decision],[num_allocate_subcarriers_lower_bound],[min_transmit_power_db],[min_number_of_URLLC_users_per_RB]]
        '''

        self.action_space = spaces.Box(low=action_space_low,high=action_space_high)
        self.observation_space = spaces.Box(low=observation_space_low, high=observation_space_high)

        self.STEP_LIMIT = 100
        self.sleep = 0
        self.steps = 0
       

    def step(self,action):
        action = np.array(action)
        #print(" ")
        #print("Action before interpolation")
        #print(action)
        action = np.transpose(action)
        #print("Action before interpolation transposed")
        #print(action)
        reward = 0
        #collect offload decisions actions 
        offload_decisions_actions = action[self.offload_decisions_label]
        #offload_decisions_actions = offload_decisions_actions[0:self.number_of_eMBB_users]
        offload_decisions_actions_mapped = []
        for offload_decision in offload_decisions_actions:
            offload_decision_mapped = interp(offload_decision,[0,1],[self.min_offload_decision,self.max_offload_decision])
            offload_decisions_actions_mapped.append(offload_decision_mapped)


        #collect subcarrier allocations actions
        
        RB_allocation_actions = action[self.allocate_num_RB_label]
       # RB_allocation_actions = RB_allocation_actions[0:self.number_of_eMBB_users]
        RB_allocation_actions_mapped = []

        for RB_allocation_action in RB_allocation_actions:
            RB_allocation_action_mapped = interp(RB_allocation_action,[0,1],[self.num_allocate_RB_lower_bound,self.num_allocate_RB_upper_bound])
            RB_allocation_actions_mapped.append(RB_allocation_action_mapped)

        RB_allocation_actions = (np.rint(RB_allocation_actions)).astype(int)
        RB_allocation_actions_mapped = (np.rint(RB_allocation_actions_mapped)).astype(int)


        #collect trasmit powers allocations actions
        transmit_power_actions = action[self.allocate_transmit_powers_label]
       # transmit_power_actions = transmit_power_actions[0:self.number_of_eMBB_users]

        transmit_power_actions_mapped = []

        for transmit_power_action in transmit_power_actions:
            transmit_power_action_mapped = interp(transmit_power_action,[0,1],[self.min_transmit_power_db,self.max_transmit_power_db])
            transmit_power_actions_mapped.append(transmit_power_action_mapped)

        #print('Action after interpolation transposed')
        #offload_decisions_actions_mapped = [1, 1, 1, 1, 1, 1, 1]
        #transmit_power_actions_mapped = [20,20,20,20,20,20,20]
        #subcarrier_allocation_actions_mapped = [10,10,10,10,10,10,10]
        #number_URLLC_Users_per_RB_action_mapped = 3
        #print("New Timestep: ", self.steps)
        print("offload_decisions_actions")
        print(offload_decisions_actions_mapped)
        print("RB_allocation_actions")
        print(RB_allocation_actions_mapped)
        print("transmit_power_actions")
        print(transmit_power_actions_mapped)


        #Perform Actions
        self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions_mapped)
        #self.SBS1.allocate_transmit_powers(self.eMBB_Users,transmit_power_actions)

        self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions_mapped)
        #self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users,offload_decisions_actions)

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS1)
        self.Communication_Channel_1.initiate_RBs()
        self.Communication_Channel_1.allocate_RBs_eMBB(self.eMBB_Users,RB_allocation_actions_mapped)

        for eMBB_User in self.eMBB_Users:
            eMBB_User.split_packet()

        for eMBB_User in self.eMBB_Users:
            if eMBB_User.has_transmitted_this_time_slot == True:
                eMBB_User.transmit_to_SBS(self.Communication_Channel_1)
                eMBB_User.local_processing()
                eMBB_User.offloading()
                eMBB_User.total_energy_consumed()
                eMBB_User.total_processing_delay()

        self.SBS1.receive_offload_packets(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_energy_consumption(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_processing_delay(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_rate_eMBB_users(self.eMBB_Users)
        self.SBS1.calculate_achieved_system_energy_efficiency()
        system_reward, reward = self.SBS1.calculate_achieved_system_reward(self.eMBB_Users)
        #print('Reward')
       # print(reward)
        #print(' ')

        #Update game state after performing actions
        for eMBB_User in self.eMBB_Users:
            eMBB_User.calculate_distance_from_SBS(self.SBS1.x_position, self.SBS1.y_position)
            eMBB_User.move_user(ENV_WIDTH,ENV_HEIGHT,self.Communication_Channel_1.long_TTI)
            eMBB_User.calculate_channel_gain()
            eMBB_User.generate_task(self.Communication_Channel_1.long_TTI)
            eMBB_User.collect_state()

        observation = np.array(self.SBS1.collect_state_space(self.eMBB_Users), dtype=np.float32)
        #print('Observation before interpolation')
        #print(np.transpose(observation))
        #normalize observation values to a range between 0 and 1 using interpolation
        row = 0
        col = 0
        min_value = 0
        max_value = 0
        for observation_type in observation:
            if row == self.OS_channel_gain_label:
                min_value = self.channel_gain_min
                max_value = self.channel_gain_max

            elif row == self.OS_comm_queue_label:
                min_value = self.communication_queue_min
                max_value = self.communication_queue_max

            elif row == self.OS_energy_harvested_label:
                min_value = self.energy_harvested_min
                max_value = self.energy_harvested_max

            elif row == self.OS_latency_label:
                min_value = self.latency_requirement_min
                max_value = self.latency_requirement_max

            elif row == self.OS_reliability_label:
                min_value = self.reliability_requirement_min
                max_value = self.reliability_requirement_max

            col = 0
            for user in observation_type:
                observation[row][col] = interp(observation[row][col],[min_value,max_value],[0,1])
                col += 1
            
            row += 1

        observation = np.transpose(observation)
        #print('observation interpolated')
        #print(observation)

        done = self.check_timestep()
        dones = [0 for element in range(len(self.eMBB_Users) - 1)]
        dones.append(done)
        info = {'reward': reward}
        self.steps+=1
        #print("reward: ", reward, "dones: ", dones)
        #print('Timestep: ', self.steps)
        return observation,reward,dones,info
    
    def reset(self):
        self.steps = 0
        self.SBS1.set_properties()
        self.OS_channel_gain_label = 0
        self.OS_comm_queue_label = 1
        self.OS_energy_harvested_label = 2
        self.OS_latency_label = 3
        self.OS_reliability_label = 4

        #Observation Space Bound Parameters
        self.channel_gain_min = self.eMBB_Users[0].min_channel_gain
        self.channel_gain_max = self.eMBB_Users[0].max_channel_gain
        self.communication_queue_min = self.eMBB_Users[0].min_communication_qeueu_size
        self.communication_queue_max = self.eMBB_Users[0].max_communication_qeueu_size
        self.energy_harvested_min = 0
        self.energy_harvested_max = self.eMBB_Users[0].max_energy_harvested
        self.latency_requirement_min = 0
        self.latency_requirement_max = self.eMBB_Users[0].max_allowable_latency
        self.reliability_requirement_min = self.eMBB_Users[0].min_allowable_reliability
        self.reliability_requirement_max = self.eMBB_Users[0].max_allowable_reliability

        for eMBB_User in self.eMBB_Users:
            eMBB_User.set_properties_UE()
            eMBB_User.set_properties_eMBB(eMBB_User.x_position,eMBB_User.y_position)

        #self.eMBB_Users.clear()

        self.SBS1.associate_users(self.eMBB_Users)
        self.Communication_Channel_1.set_properties()

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS1)
        self.Communication_Channel_1.initiate_RBs()
        info = {'reward': 0}
        self.SBS1.collect_state_space(self.eMBB_Users)
        observation = np.array(self.SBS1.system_state_space, dtype=np.float32)
        #print('Observation before transpose')
        #print(np.transpose(observation))
        #normalize observation values to a range between 0 and 1 using interpolation
        row = 0
        col = 0
        min_value = 0
        max_value = 0
        for observation_type in observation:
            if row == self.OS_channel_gain_label:
                min_value = self.channel_gain_min
                max_value = self.channel_gain_max

            elif row == self.OS_comm_queue_label:
                min_value = self.communication_queue_min
                max_value = self.communication_queue_max

            elif row == self.OS_energy_harvested_label:
                min_value = self.energy_harvested_min
                max_value = self.energy_harvested_max

            elif row == self.OS_latency_label:
                min_value = self.latency_requirement_min
                max_value = self.latency_requirement_max

            elif row == self.OS_reliability_label:
                min_value = self.reliability_requirement_min
                max_value = self.reliability_requirement_max

            col = 0
            for user in observation_type:
                observation[row][col] = interp(observation[row][col],[min_value,max_value],[0,1])
                col += 1
            
            row += 1
        observation = np.transpose(observation)
        #print('observation interpolated')
        #print(observation)
        reward = 0
        done = 0
        return observation

    def render(self, mode='human'):
        pass

    def create_objects(self):
        #Small Cell Base station
        self.SBS1 = SBS(ENV_WIDTH/2,ENV_HEIGHT/2)

        #Users
        #Group all eMBB users
        self.eMBB_Users = []
        number_of_users = 7
        for i in range(1,number_of_users+1):
            x_pos = random.randint(0,ENV_WIDTH)
            y_pos = random.randint(0,ENV_HEIGHT)
            self.eMBB_Users.append(eMBB_UE(i,x_pos,y_pos))

        #Communication Channel
        self.Communication_Channel_1 = Communication_Channel()

        #Associate SBS with users
        self.SBS1.associate_users(self.eMBB_Users)

    def check_timestep(self):
        if self.steps >= self.STEP_LIMIT:
            return True
        else: 
            return False
        
    def seed(self):
        pass