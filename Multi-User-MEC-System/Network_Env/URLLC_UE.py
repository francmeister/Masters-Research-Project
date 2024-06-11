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

class URLLC_UE(User_Equipment):
    def __init__(self, URLLC_UE_label,User_label, x,y):
        #User_Equipment.__init__(self)
        self.UE_label = URLLC_UE_label
        #Urllc users will be tagged with a 1
        self.user_label = User_label
        self.type_of_user_id = 1
        self.original_x_position = x
        self.original_y_position = y
        self.URLLC_UE_label = URLLC_UE_label
        self.communication_channel = Communication_Channel(1)
        self.assigned_access_point = 0
        self.assigned_access_point_label_matrix = []
        self.assigned_access_point_label_matrix_integers = []
        self.set_properties_URLLC()

    def set_properties_URLLC(self):
        self.max_task_arrival_rate_tasks_per_second = 10
        self.min_task_arrival_rate_tasks_per_second = 5
        self.cycles_per_byte = 330
        self.cycles_per_bit = self.cycles_per_byte/8
        self.max_service_rate_cycles_per_slot = 620000
        self.service_rate_bits_per_slot = (self.max_service_rate_cycles_per_slot/self.cycles_per_byte)*8
        self.energy_consumption_coefficient = math.pow(10,-13.8)
        self.total_gain = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound*2)
        self.total_gain_on_allocated_rb = 0
        self.small_scale_gain = 0
        self.small_scale_gain_on_allocated_rb = 0
        self.large_scale_gain_on_allocated_rb = 0
        self.large_scale_gain = 0
        self.achieved_channel_rate = 0
        self.task_arrival_rate_tasks_per_slot = 0
        self.task_arrival_rate_tasks_per_slot = 0
        self.timeslot_counter = 0
        self.task_identifier = 0
        self.task_queue = []
        self.local_task_queue = []
        self.offload_task_queue = []
        self.small_scale_channel_gain_threshold = 0
        self.task_size_per_slot_bits = 300
        self.latency_requirement = 1#latency required is 10 ms for every task#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
        self.reliability_requirement = 0
        self.assigned_resource_block = 0
        self.assigned_time_block = 0
        self.assigned_resource_time_block = []
        self.puncturing_embb_user_transmit_power = 0
        self.puncturing_embb_user_small_scale_gain = 0
        self.puncturing_embb_user_large_scale_gain = 0
        self.transmit_power = 400
        self.achieved_channel_rate = 0
        

    def generate_task(self,communication_channel):
        self.has_transmitted_this_time_slot = False
        self.timeslot_counter+=1

        #Specify slot task size, computation cycles and latency requirement
        #self.task_arrival_rate_tasks_per_second = random.randint(self.min_task_arrival_rate_tasks_per_second,self.max_task_arrival_rate_tasks_per_second)
        self.task_arrival_rate_tasks_per_slot = np.random.binomial(size=1,n=1,p=1)#np.random.poisson(5,1)
        self.task_arrival_rate_tasks_per_slot = self.task_arrival_rate_tasks_per_slot[0]
        #self.task_size_per_slot_bits = 10#10 bits per task in slot 
        qeueu_timer = 0
        
        if self.task_arrival_rate_tasks_per_slot == 1:
            QOS_requirement_ = QOS_requirement(self.latency_requirement,self.reliability_requirement)
            user_task = Task(330,self.task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
            self.task_identifier+=1
            self.task_queue.append(user_task)

    def calculate_channel_gain_on_all_resource_blocks(self,communication_channel):
        #Pathloss gain
        #self.pathloss_gain = (math.pow(10,(35.3+37.6*math.log10(self.distance_from_SBS))))/10
        number_of_RBs = communication_channel.num_allocate_RBs_upper_bound
        small_scale_gain = np.random.exponential(1,size=(1,number_of_RBs))
        large_scale_gain = np.random.exponential(1,size=(1,number_of_RBs))
        self.small_scale_channel_gain = small_scale_gain
        first_large_scale_gain = large_scale_gain[0][0]
        item = 0
        for gain in large_scale_gain[0]:
            large_scale_gain[0][item] = first_large_scale_gain
            item+=1

        self.small_scale_gain = small_scale_gain
        self.large_scale_gain = large_scale_gain
        self.small_scale_gain_on_allocated_rb = self.small_scale_gain[0][self.assigned_resource_block-1]
        self.large_scale_gain_on_allocated_rb = self.large_scale_gain[0][self.assigned_resource_block-1]
        #print('small_scale_gain')
        #print(small_scale_gain)
        #print('larger_scale_gain')
        #print(large_scale_gain)
        #self.total_gain = np.concatenate((small_scale_gain,large_scale_gain),axis=1)#np.random.exponential(1,size=(1,number_of_RBs))
        self.total_gain_on_allocated_rb = self.small_scale_gain_on_allocated_rb*self.large_scale_gain_on_allocated_rb
        self.total_gain = small_scale_gain*large_scale_gain
        self.total_gain = self.total_gain.squeeze()
        

        #print('self.total_gain')
        #print(self.total_gain)
        #self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        #self.total_gain = self.small_scale_channel_gain#*self.large_scale_channel_gain#self.pathloss_gain
        #if self.total_gain < 0.1:
        #    self.total_gain = 0.1
    #def calculate_channel_gain_on_assigned_resource_block(self,communication_channel):

        #self.small_scale_gain = np.random.exponential(1,size=1)
        #self.large_scale_gain = np.random.exponential(1,size=1)
        #self.total_gain = self.small_scale_gain*self.large_scale_gain

    def split_tasks(self):
        # if self.small_scale_gain_on_allocated_rb < self.small_scale_channel_gain_threshold and len(self.task_queue) > 0:
        #     self.local_task_queue.append(self.task_queue[0])
        #     self.task_queue.clear()
        # elif self.small_scale_gain_on_allocated_rb > self.small_scale_channel_gain_threshold and len(self.task_queue) > 0:
        if len(self.task_queue) > 0:
            self.offload_task_queue.append(self.task_queue[0])
            self.has_transmitted_this_time_slot = True
            self.task_queue.clear()

    def find_puncturing_embb_users(self,eMBB_users):
        self.puncturing_embb_user_transmit_power = 0
        self.puncturing_embb_user_small_scale_gain = 0
        self.puncturing_embb_user_large_scale_gain = 0
        punturing_embb_user = 0

        for eMBB_user in eMBB_users:
            for allocated_rb in eMBB_user.allocated_resource_blocks_numbered:
                if allocated_rb == self.assigned_resource_block:
                    if eMBB_user.time_matrix[allocated_rb-1] == 1 or eMBB_user.time_matrix[allocated_rb-1] == 2:
                        if eMBB_user.time_matrix[allocated_rb-1] == self.assigned_time_block:
                            #print('self.assigned_resource_block: ', self.assigned_resource_block, "self.assigned_time_block: ", self.assigned_time_block)
                            #print('eMBB_user.eMBB_UE_label: ', eMBB_user.eMBB_UE_label, 'eMBB_user.time_matrix: ', eMBB_user.time_matrix, "allocated_rb: ", allocated_rb, "time_blocks: ", eMBB_user.time_matrix[allocated_rb-1])
                            self.puncturing_embb_user_transmit_power = eMBB_user.assigned_transmit_power_W
                            punturing_embb_user = eMBB_user.eMBB_UE_label
                            # print('self.assigned_resource_block: ', self.assigned_resource_block)
                            # print('eMBB_user.small_scale_gain: ', eMBB_user.small_scale_gain)
                            self.puncturing_embb_user_small_scale_gain = eMBB_user.small_scale_gain[0][(self.assigned_resource_block-1)]
                            self.puncturing_embb_user_large_scale_gain = eMBB_user.large_scale_gain[0][(self.assigned_resource_block-1)]
                            break
                    elif eMBB_user.time_matrix[allocated_rb-1] == (1,2):
                        if eMBB_user.time_matrix[allocated_rb-1][0] == self.assigned_time_block or eMBB_user.time_matrix[allocated_rb-1][1] == self.assigned_time_block:
                            #print('self.assigned_resource_block: ', self.assigned_resource_block, "self.assigned_time_block: ", self.assigned_time_block)
                            #print('eMBB_user.eMBB_UE_label: ', eMBB_user.eMBB_UE_label, 'eMBB_user.time_matrix: ', eMBB_user.time_matrix, "allocated_rb: ", allocated_rb, "time_blocks: ", eMBB_user.time_matrix[allocated_rb-1])
                            self.puncturing_embb_user_transmit_power = eMBB_user.assigned_transmit_power_W
                            # print('self.assigned_resource_block: ', self.assigned_resource_block)
                            # print('eMBB_user.small_scale_gain: ', eMBB_user.small_scale_gain)
                            self.puncturing_embb_user_small_scale_gain = eMBB_user.small_scale_gain[0][(self.assigned_resource_block-1)]
                            self.puncturing_embb_user_large_scale_gain = eMBB_user.large_scale_gain[0][(self.assigned_resource_block-1)]
                            break
                # for time_blocks in eMBB_user.time_matrix:
                #     if time_blocks == 1 or time_blocks == 2:
                #         if allocated_rb == self.assigned_resource_block and self.assigned_time_block == time_blocks:
                #             print('self.assigned_resource_block: ', self.assigned_resource_block, "self.assigned_time_block: ", self.assigned_time_block)
                #             print('eMBB_user.eMBB_UE_label: ', eMBB_user.eMBB_UE_label, 'eMBB_user.time_matrix: ', eMBB_user.time_matrix, "allocated_rb: ", allocated_rb, "time_blocks: ", time_blocks)
                #             self.puncturing_embb_user_transmit_power = eMBB_user.assigned_transmit_power_W
                #             punturing_embb_user = eMBB_user.eMBB_UE_label
                #             # print('self.assigned_resource_block: ', self.assigned_resource_block)
                #             # print('eMBB_user.small_scale_gain: ', eMBB_user.small_scale_gain)
                #             self.puncturing_embb_user_small_scale_gain = eMBB_user.small_scale_gain[0][(self.assigned_resource_block-1)]
                #             self.puncturing_embb_user_large_scale_gain = eMBB_user.large_scale_gain[0][(self.assigned_resource_block-1)]
                #             break
                #     elif time_blocks == (1,2):
                #         for time_block in time_blocks:
                #             if allocated_rb == self.assigned_resource_block and self.assigned_time_block == time_block:
                #                 print('self.assigned_resource_block: ', self.assigned_resource_block, "self.assigned_time_block: ", self.assigned_time_block)
                #                 print('eMBB_user.eMBB_UE_label: ', eMBB_user.eMBB_UE_label, 'eMBB_user.time_matrix: ', eMBB_user.time_matrix, "allocated_rb: ", allocated_rb, "time_blocks: ", time_blocks)
                #                 self.puncturing_embb_user_transmit_power = eMBB_user.assigned_transmit_power_W
                #                 # print('self.assigned_resource_block: ', self.assigned_resource_block)
                #                 # print('eMBB_user.small_scale_gain: ', eMBB_user.small_scale_gain)
                #                 self.puncturing_embb_user_small_scale_gain = eMBB_user.small_scale_gain[0][(self.assigned_resource_block-1)]
                #                 self.puncturing_embb_user_large_scale_gain = eMBB_user.large_scale_gain[0][(self.assigned_resource_block-1)]
                #                 break
        #print('urllc user: ', self.URLLC_UE_label, "puncturing embb: ", punturing_embb_user, "self.assigned_resource_block: ", self.assigned_resource_block, "self.assigned_time_block: ", self.assigned_time_block)
        #print('')
        # print('------------------------------------')
        # print('URLLC user: ', self.URLLC_UE_label, 'assigned RB: ', self.assigned_resource_block)
        # print('URLLC user: ', self.URLLC_UE_label, 'assigned time block: ', self.assigned_time_block)
        # for eMBB_User in eMBB_users:
        #     print('eMBB user id: ', eMBB_User.UE_label)
        #     print('allocated rbs: ', eMBB_User.allocated_resource_blocks_numbered)
        #     print('allocated time block: ', eMBB_User.time_matrix)
        #     print('eMBB_user small scale gain: ', eMBB_User.small_scale_gain)
        #     print('large scale gain: ', eMBB_User.large_scale_gain)
        #     print('assigned_transmit_power_W: ', eMBB_User.assigned_transmit_power_W)
        #     print('')

        # print('')
        # print('self.puncturing_embb_user_transmit_power: ',self.puncturing_embb_user_transmit_power)
        # print('self.puncturing_embb_user_small_scale_gain: ', self.puncturing_embb_user_small_scale_gain)
        # print('self.puncturing_embb_user_large_scale_gain: ', self.puncturing_embb_user_large_scale_gain)
        # print('------------------------------------')

    def calculate_achieved_channel_rate(self,eMBB_users,communication_channel):
        self.find_puncturing_embb_users(eMBB_users)
        numerator = self.small_scale_gain[0][self.assigned_resource_block-1]*self.large_scale_gain[0][self.assigned_resource_block-1]*self.transmit_power
        denominator = self.puncturing_embb_user_large_scale_gain*self.puncturing_embb_user_small_scale_gain*self.puncturing_embb_user_transmit_power + communication_channel.noise_spectral_density_W 
        channel_rate = communication_channel.RB_bandwidth_Hz*(1/communication_channel.num_of_mini_slots)*math.log2((1+numerator/denominator))
        #self.achieved_channel_rate = channel_rate/500
        self.achieved_channel_rate = channel_rate/1000
        #print('urllc user id: ', self.URLLC_UE_label, 'achieved channel rate: ', self.achieved_channel_rate)



    # def transmit_to_SBS(self, communication_channel):
    #     if self.has_transmitted_this_time_slot == True:
    #         achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel,RB_indicator,RB_channel_gain)
    #         min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
    #         self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,7000],[0,1])   

    # def calculate_channel_rate(self, communication_channel,RB_indicator,RB_channel_gain):
    #     RB_bandwidth = communication_channel.RB_bandwidth_Hz
    #     noise_spectral_density = communication_channel.noise_spectral_density_W
    #     channel_rate_numerator = self.assigned_transmit_power_W*RB_channel_gain
    #     channel_rate_denominator = noise_spectral_density#*RB_bandwidth
    #     channel_rate = RB_indicator*(RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator)))
    #     return (channel_rate/1000)
    def initial_large_scale_gain_all_access_points(self,num_access_point):
        large_scale_gain = np.random.exponential(1,size=(1,num_access_point))
        large_scale_gain = large_scale_gain.squeeze()
        return large_scale_gain
    
    def initial_arrival_rates(self):
        task_arrival_rate = np.random.poisson(25,1)
        task_arrival_rate = task_arrival_rate[0]
        return task_arrival_rate
    

    def assigned_access_point_label_matrix_to_numpy_array(self):
        #self.assigned_access_point_label_matrix = np.array(self.assigned_access_point_label_matrix)
       
        count = 0
        index = 0
        for assigned_access_point_label in self.assigned_access_point_label_matrix:
            index = np.where(assigned_access_point_label == 1)[0]
            self.assigned_access_point_label_matrix_integers.append(index+1)

        self.assigned_access_point_label_matrix_integers = np.array(self.assigned_access_point_label_matrix_integers)



