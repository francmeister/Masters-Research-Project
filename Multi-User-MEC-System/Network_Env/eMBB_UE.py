import pygame, sys, time, random
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
from scipy.stats import poisson, nbinom

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,User_label,x,y):
        #User_Equipment.__init__(self)
        #embb users will be tagged with a 0
        self.user_label = User_label
        self.type_of_user_id = 0
        self.UE_label = eMBB_UE_label
        self.original_x_position = x
        self.original_y_position = y
        self.eMBB_UE_label = eMBB_UE_label
        self.communication_channel = Communication_Channel(1)
        self.assigned_access_point = 0
        self.assigned_access_point_label_matrix = []
        self.assigned_access_point_label_matrix_integers = []
        self.timestep_counter = 0
        self.cycles_per_byte = 330
        self.cycles_per_bit = self.cycles_per_byte/8
        self.local_queueing_violation_prob_reward = 0
        self.offload_queueing_violation_prob_reward = 0
        self.offload_time_delay_violation_prob_constraint_violation_count = 0
        self.offloa_ratio_reward = 0
        self.available_resource_time_code_block = []
        self.max_queue_length = 0
        self.max_queue_length_bits = 0
        self.rho = 0
        #self.max_service_rate_cycles_per_slot = random.randint(5000,650000)#620000
        #self.max_service_rate_cycles_per_slot = 620000
        #self.service_rate_bits_per_second = 2500000 #2.5MB/s(random.randint(5,5000))
        self.service_rate_bits_per_second = 969700    #121212.121212#120000#random.randint(100000,300000)#120000
        self.service_rate_bits_per_slot = self.service_rate_bits_per_second/1000 
        self.max_service_rate_cycles_per_slot = self.service_rate_bits_per_slot*self.cycles_per_bit
        self.max_bits_process_per_slot = self.max_service_rate_cycles_per_slot/self.cycles_per_bit
        #print('self.max_service_rate_cycles_per_slot: ', self.max_service_rate_cycles_per_slot)
        #self.service_rate_bits_per_slot = (self.max_service_rate_cycles_per_slot/self.cycles_per_byte)*8
        self.local_queue_length = 0
        self.offload_queue_length = 0

        self.local_queue_delay =0
        self.offload_queue_delay = 0
        self.channel_gain_scaling_factor = 1
        self.x_coordinate = np.random.uniform(low=30, high=100)
        self.y_coordinate = np.random.uniform(low=30, high=100)
        self.distance_from_SBS_ = np.random.uniform(low=30, high=100)
        self.average_offloading_rate = 0

        self.battery_energy_level_ = 0
        self.small_scale_gain_ = 0
        self.large_scale_gain_ = 0
        self.com_queue_length = 0
        self.loc_queue_length = 0
        self.packet_size_bits = 100 # 12000 bits per packet
        self.cycles_per_packet = self.packet_size_bits*self.cycles_per_bit
        self.max_allowable_latency_ = 2000
        self.local_queue_delay_violation_probability_constraint = 0.2
        self.num_of_clustered_urllc_users = 0
        self.task_arrival_rate_multiplier = 0.25
        self.average_task_size = 600#200
        self.geometric_probability = 1/self.average_task_size
        self.task_arrival_rate_tasks_per_second = 0
        self.average_task_arrival_rate = 5
        self.Pr_Qs = []
        self.average_data_rate = 0
        self.Ld_max = 15
        
        #num_puncturing_users

        self.calculate_offloading_rate()


        self.set_properties_eMBB()

    def set_properties_eMBB(self):
        self.average_data_rate = 0
        self.queuing_latency = 0
        self.local_queueing_latency = 0
        self.offload_queueing_latency = 0
        self.offlaod_traffic_numerator = 0
        self.offload_stability_constraint_reward = 0
        #State Space Limits
        self.timestep_counter_ = 0
        self.max_allowable_latency = 2000 #[1,2] s
        self.min_allowable_latency = 1000

        self.max_allowable_reliability = 0

        self.min_communication_qeueu_size = 0
        self.max_communication_qeueu_size = 100

        self.min_channel_gain = math.pow(10,-8)
        self.max_channel_gain = 25

        self.min_energy_harvested = 0
        self.max_energy_harvested = 150

        self.max_battery_energy = 2500#22000
        self.min_battery_energy = 0

        self.max_cpu_frequency = 5000
        self.min_cpu_frequency = 5

        self.max_task_size_KB_per_second = 100 #100KB per second
        self.min_task_size_KB_per_second = 50 #50KB per second

        self.max_queue_length_KBs = math.pow(self.max_task_size_KB_per_second,3) # 1GB
        self.min_queue_length_KBs = 0

        self.max_task_arrival_rate_tasks_per_second = 10
        self.min_task_arrival_rate_tasks_per_second = 5

        self.max_queue_length_number = self.calculate_max_queue_length_number(self.communication_channel,self.max_task_arrival_rate_tasks_per_second)
        #print('self.max_queue_length_number: ', self.max_queue_length_number)
        self.min_queue_length = 0

        self.max_lc_queue_length = 300
        self.max_off_queue_length = 300

        self.min_lc_queue_length = 0
        self.min_off_queue_length = 0

        self.battery_energy_level = 100#(random.randint(15000,25000))
        self.energy_harvesting_constant = 5

        self.battery_energy_level_sim = self.battery_energy_level
        self.energy_harvested_sim = 0

        self.local_queue_length_num_tasks = 0
        self.offload_queue_length_num_tasks = 0
        

        #self.QOS_requirement = QOS_requirement()
        #self.QOS_requirement_for_transmission = QOS_requirement()
        #self.user_task = Task(330)
        #self.local_task = Task(330)
        #self.offload_task = Task(330)
        self.expected_rate_over_prev_T_slot = 0
        self.average_task_size_offload_queue = 0
        self.local_computation_delay_seconds = 0
        self.achieved_local_energy_consumption = 0
        self.offload_transmission_energy = 0
        #self.battery_energy_level = 100 # Begin with 100%
        self.energy_harvested = 0
        self.achieved_transmission_delay = 0
        self.local_queue = []
        self.timeslot_counter = 0
        self.x_position = self.original_x_position
        self.y_position = self.original_y_position
        self.energy_harversted = 0
        self.distance_from_SBS = 0
        self.has_transmitted_this_time_slot = False
        self.communication_queue = []
        #self.energy_consumption_coefficient = math.pow(10,-12.3)
        self.energy_consumption_coefficient = math.pow(10,-18)
        self.achieved_transmission_energy_consumption = 0
        self.achieved_local_processing_delay = 0
        self.achieved_total_energy_consumption = 0
        self.achieved_total_processing_delay = 0
        self.cpu_cycles_per_byte = 330
        self.cpu_clock_frequency = (random.randint(5,5000)) #cycles/slot
        self.user_state_space = State_Space()
        self.allocated_offloading_ratio = 0
        self.packet_offload_size_bits = 0
        self.packet_local_size_bits = 0
        self.packet_size = 0
        self.delay_reward = 10
        self.achieved_channel_rate_normalized = 0
        self.achieved_total_energy_consumption_normalized = 0
        self.dequeued_local_tasks = []
        self.dequeued_offload_tasks = []
        self.completed_tasks = []
   
    
        self.single_side_standard_deviation_pos = 5
        self.xpos_move_lower_bound = self.x_position - self.single_side_standard_deviation_pos
        self.xpos_move_upper_bound = self.x_position + self.single_side_standard_deviation_pos
        self.ypos_move_lower_bound = self.y_position - self.single_side_standard_deviation_pos
        self.ypos_move_upper_bound = self.y_position + self.single_side_standard_deviation_pos
        self.allocated_RBs = []

        self.max_transmission_power_dBm = 40 # dBm
        self.min_transmission_power_dBm = 0
        self.max_transmission_power_W =  (math.pow(10,(self.max_transmission_power_dBm/10)))/1000# Watts
        self.min_transmission_power_W =  (math.pow(10,(self.min_transmission_power_dBm/10)))/1000# Watts
        self.assigned_transmit_power_dBm = 0
        self.assigned_transmit_power_W = 0
        
       
        self.small_scale_channel_gain = 0
        self.large_scale_channel_gain = 0
        self.pathloss_gain = 0
        self.achieved_channel_rate = 0
        self.allowable_latency = 0
        self.task_queue = []
        self.task_identifier = 0
        self.ptr = 0
        self.queuing_delay = 0
        self.previous_slot_battery_energy = 0

        self.total_gain = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound*2)
        self.total_gain_ = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound*2)
        self.previous_arrival_rate = 0
        self.previous_arrival_rate_off = 0
        self.previous_arrival_rate_lc = 0
        self.previous_service_rate_off = 0
        self.previous_service_rate_lc = 0
        self.previous_traffic_intensity_off = 0
        self.previous_traffic_intensity_lc = 0
        self.previous_channel_rate = 0
        self.previous_offloading_ratio = 0
        self.previous_channel_rate = 0
        self.previous_task_size_bits = 0
        self.current_queue_length_off = 0
        self.current_queue_length_lc = 0
        self.current_arrival_rate = 0
        self.current_queue_length_modified_lc = 0
        self.current_queue_length_modified_off = 0
        self.tasks_dropped = 0
        self.small_scale_gain = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound)
        self.small_scale_gain = np.array([self.small_scale_gain])
        self.large_scale_gain = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound)
        self.large_scale_gain = np.array([self.large_scale_gain])
        self.communication_queue_size_before_offloading = 0
        self.allocated_resource_blocks_numbered = []
        self.time_allocators = []
        self.time_matrix = []
        self.puncturing_urllc_users_ = []
        self.occupied_resource_time_blocks = []
        self.achieved_channel_rate_ = 0
        self.previous_rates = []
        self.ptr = 0
        self.previous_rates_ = []
        self.pointer_ = 0
        self.task_arrival_rate = 0
        self.offloading_ratio = 0
        self.average_packet_size_bits = 0
        self.max_lc_queue_delay_violation_probability = 0.8
        self.completed_tasks = []
        self.completed_tasks_ = []

        self.local_queue_lengths = []
        self.offload_queue_lengths = []

        self.local_queue_delays = []
        self.offload_queue_delays = []

        self.ptr_local_queue_lengths = 0


        self.ptr_offload_queue_length = 0

        self.local_delays = []
        self.ptr_local_delay = 0

        self.offload_delays = []
        self.ptr_offload_delay = 0
        self.average_local_queue_length=0
        self.average_offload_queue_length=0
        self.average_local_delays=0
        self.average_offload_delays=0
        self.episode_energy = 0
        self.times_to_generate_tasks = []
        self.energy_conversion_efficiency = 1
        self.BS_transmit_power = 5 # 5W
        self.pathloss_coefficient = 1.5
        self.antenna_gain = 10 # 10dB
        self.slot_time_ms = 10**(-3)
        self.max_battery_capacity = 26640
        self.battery_energy_level = self.max_battery_capacity#(random.randint(15000,25000))
        self.energy_harvesting_constant = 5
        self.numbers_of_puncturing_users = 0
        self.number_of_allocated_RBs = 0
        self.local_traffic_intensity = 0

        self.battery_energy_constraint_violation_count = 0
        self.local_queueing_traffic_constraint_violation_count = 0
        self.offload_queueing_traffic_constaint_violation_count = 0
        self.local_time_delay_violation_prob_constraint_violation_count = 0
        self.rmin_constraint_violation_count = 0
        self.average_bits_tasks_arriving = 200
        self.local_queue_delay_violation_probability_ = 0
        self.offload_queue_delay_violation_probability_ = 0

        #----------------------variables for queueing delay violation probability calculation-------------------------------------
        self.Ld_lc = 0 #Computed in self.split_task() function
        self.Qd_lc = 0 #Computed in self.split_task() function
        self.computation_time_per_bit = self.cycles_per_bit/self.max_service_rate_cycles_per_slot
        self.T_max_lc = 0.01
        #self.Ld_max = round(self.T_max_lc/self.computation_time_per_bit)
        self.offload_queue_delay_ = 0
        self.local_queue_delay_ = 0
        # print('self.computation_time_per_bit: ', self.computation_time_per_bit)
        # print('self.T_max_lc: ', self.T_max_lc)
        # print('self.Ld_max: ', self.Ld_max)

        #self.large_scale_gain_
    
  


    def move_user(self,ENV_WIDTH,ENV_HEIGHT):
        self.x_position = random.randint(self.xpos_move_lower_bound,self.xpos_move_upper_bound)
        self.y_position = random.randint(self.ypos_move_lower_bound,self.ypos_move_upper_bound)

        if self.x_position < 0 or self.x_position > ENV_WIDTH:
            self.x_position = self.original_x_position

        if self.y_position < 0 or self.y_position > ENV_HEIGHT:
            self.y_position = self.original_y_position
        

    def calculate_max_queue_length_number(self,communication_channel,max_task_arrival_rate_tasks_per_second):
        max_task_size_per_second_kilobytes = self.max_task_size_KB_per_second # 100 kilobytes
        max_task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*max_task_arrival_rate_tasks_per_second
        max_task_size_per_slot_kilobytes = int(max_task_size_per_second_kilobytes*max_task_arrival_rate_tasks_slot)

        return (self.max_queue_length_KBs/max_task_size_per_slot_kilobytes)

    def generate_task(self,communication_channel):
        self.has_transmitted_this_time_slot = False
        self.timeslot_counter+=1

        #Specify slot task size, computation cycles and latency requirement
        #self.task_arrival_rate_tasks_per_second = random.randint(self.min_task_arrival_rate_tasks_per_second,self.max_task_arrival_rate_tasks_per_second)
        self.task_arrival_rate_tasks_per_second = np.random.poisson(self.average_task_arrival_rate,1)#np.random.poisson(25,1)#np.random.poisson(5,1)
        self.task_arrival_rate_tasks_per_second = self.task_arrival_rate_tasks_per_second[0]
        self.task_arrival_rate = self.task_arrival_rate_tasks_per_second
        self.previous_arrival_rate = self.task_arrival_rate_tasks_per_second
        self.current_arrival_rate = self.task_arrival_rate_tasks_per_second
        qeueu_timer = 0
    

        # if len(self.task_queue) >= self.max_queue_length_number:
        #     for x in range(0,self.task_arrival_rate_tasks_per_second):
        #         #np.random.poisson(10)
        #         #task_size_per_second_kilobytes = random.randint(self.min_task_size_KB_per_second,self.max_task_size_KB_per_second) #choose between 50 and 100 kilobytes
        #         #task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*self.task_arrival_rate_tasks_per_second
        #         #task_size_per_slot_kilobytes = task_size_per_second_kilobytes*task_arrival_rate_tasks_slot
        #         #task_size_per_slot_bits = int(np.random.uniform(400,800))#int(np.random.uniform(500,1500))#Average of 1000 bits per task in slot #int(task_size_per_slot_kilobytes*8000) #8000 bits in a KB----------
        #         task_size_per_slot_bits = int(np.random.geometric(self.geometric_probability))
        #         #task_size_per_slot_bits = self.average_task_size
        #         #task_size_per_slot_bits = int(np.random.uniform(2000,5000))
        #         #task_size_per_slot_bits = 1000#int(np.random.uniform(400,800))
        #         self.packet_size_bits = 100 # 12000 bits per packet
        #         self.cycles_per_packet = self.packet_size_bits*self.cycles_per_bit
        #         self.previous_task_size_bits = task_size_per_slot_bits
        #         #task_cycles_required = self.cycles_per_bit*task_size_per_slot_bits#-------------
        #         latency_requirement = 10#latency required is 10 ms for every task#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
        #         reliability_requirement = 0
        #         QOS_requirement_ = QOS_requirement(latency_requirement,reliability_requirement)
        #         user_task = Task(330,task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
        #         self.task_identifier+=1
        #         #print('task identifier: ', self.task_identifier)

        #         self.storage[int(self.ptr)] = user_task
        #         self.ptr = (self.ptr + 1) % self.max_queue_length_number
        # else:
        for x in range(0,self.task_arrival_rate_tasks_per_second):
            #task_size_per_second_kilobytes = random.randint(self.min_task_size_KB_per_second,self.max_task_size_KB_per_second) #choose between 50 and 100 kilobytes
            #task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*self.task_arrival_rate_tasks_per_second
            #task_size_per_slot_kilobytes = task_size_per_second_kilobytes*task_arrival_rate_tasks_slot
            #task_size_per_slot_bits = self.average_task_size
            task_size_per_slot_bits = int(np.random.geometric(self.geometric_probability))#int(np.random.uniform(500,1500)) #8000 bits in a KB----------
            #task_size_per_slot_bits = int(np.random.uniform(2000,5000))
            #task_size_per_slot_bits = 1000#int(np.random.uniform(400,800))
            self.packet_size_bits = 100 # 12000 bits per packet
            self.cycles_per_packet = self.packet_size_bits*self.cycles_per_bit
            self.previous_task_size_bits = task_size_per_slot_bits
            #task_cycles_required = self.cycles_per_bit*task_size_per_slot_bits#-------------
            latency_requirement = 10#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
            reliability_requirement = 0
            QOS_requirement_ = QOS_requirement(latency_requirement,reliability_requirement)
            user_task = Task(330,task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
            self.task_identifier+=1
            #print('task identifier: ', self.task_identifier)
            self.task_queue.append(user_task)
        

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos):

        x_diff_metres = abs(SBS_x_pos-self.x_coordinate)
        y_diff_metres = abs(SBS_y_pos-self.y_coordinate)


        self.distance_from_SBS_ = math.sqrt(math.pow(x_diff_metres,2)+math.pow(y_diff_metres,2))

    def collect_state(self):
        #self.cpu_clock_frequency = (random.randint(5,5000))
        offloading_queue_length, local_queue_length = self.calculate_queue_lengths()
        self.user_state_space.collect(self.total_gain_,self.previous_slot_battery_energy,len(self.communication_queue), len(self.local_queue))
        #self.user_state_space.collect(self.total_gain,self.communication_queue,self.battery_energy_level,self.communication_queue[0].QOS_requirement,self.cpu_clock_frequency)
        return self.user_state_space

    def split_tasks(self):

        self.local_queue_length_num_tasks = 0
        self.offload_queue_length_num_tasks = 0
        self.average_local_queue_length = 0
        self.average_offload_queue_length = 0
        
        if len(self.task_queue) > 0:
            local_bits = 0
            offloading_bits = 0
            total_bits = 0
            task_identities = []
            task_sizes_bits = []
            required_cycles = []
            latency_requirements = []

            if len(self.task_queue) > 0:
                for task in self.task_queue:
                    task_identities.append(task.task_identifier)
                    latency_requirements.append(task.QOS_requirement.max_allowable_latency)
                    task_sizes_bits.append(task.slot_task_size)
                    required_cycles.append(task.required_computation_cycles)
                    self.average_bits_tasks_arriving+=task.slot_task_size
            self.average_bits_tasks_arriving = self.average_bits_tasks_arriving/len(self.task_queue)
            self.average_bits_tasks_arriving = self.average_task_size
            data = {
                'Task Identity':task_identities,
                'Task Size Bits':task_sizes_bits,
                #'Required Cycles':required_cycles,
                'Latency requirement':latency_requirements
            }
            #print('Before Processing**************************************************')
            df = pd.DataFrame(data=data)
            #print('--------------------------------------------Timeslot: ',self.timeslot_counter, '--------------------------------------------')
            #print('Before Processing**************************************************')
            # print('task queue data')
            # print(df)
            #print(' ')
            #self.allocated_offloading_ratio = 0.8
            # print('self.allocated_offloading_ratio')
            # print(self.allocated_offloading_ratio)
           # print('')
            for x in range(0,len(self.task_queue)):
                packet_dec = self.task_queue[x].bits
                packet_dec = self.task_queue[x].slot_task_size
                #print('Task size: ', packet_dec)
                #print('packet_dec: ', packet_dec)
                total_bits+=self.task_queue[x].slot_task_size
                self.QOS_requirement_for_transmission = self.task_queue[x].QOS_requirement
                packet_bin = bin(packet_dec)[2:]
                #print('packet_bin: ', packet_bin)
                packet_size = len(packet_bin)
                #print('packet_size: ', packet_size)
                
                # print('packet_size: ', packet_size)
                # print('self.allocated_offloading_ratio*packet_size:', self.allocated_offloading_ratio*packet_size)
                #self.packet_offload_size_bits = round(self.allocated_offloading_ratio*packet_size)
                #print('self.packet_offload_size_bits: ', self.allocated_offloading_ratio*packet_dec)
                self.packet_offload_size_bits = round(self.allocated_offloading_ratio*packet_dec)
                #self.packet_local_size_bits =  packet_size - self.packet_offload_size_bits
                #print('1-self.allocated_offloading_ratio)*packet_dec: ', (1-self.allocated_offloading_ratio)*packet_dec)
                self.packet_local_size_bits = round((1-self.allocated_offloading_ratio)*packet_dec)
        
                #print('')
                local_bits+=self.packet_local_size_bits
                offloading_bits+=self.packet_offload_size_bits

                if self.packet_local_size_bits > 0:
                    local_task = Task(330,self.packet_local_size_bits,self.task_queue[x].QOS_requirement,self.task_queue[x].queue_timer,self.task_queue[x].task_identifier)
                    self.local_queue.append(local_task)

                if self.packet_offload_size_bits > 0:
                    offload_task = Task(330,self.packet_offload_size_bits,self.task_queue[x].QOS_requirement,self.task_queue[x].queue_timer,self.task_queue[x].task_identifier)
                    self.communication_queue.append(offload_task)

            #print('-------------------------------------------------------------------------------------------------------------------')
            # print('Number of tasks generated: ', len(self.task_queue), 'Total Size bits: ', total_bits)
            # print('Offloading Ratio: ', self.allocated_offloading_ratio)
            # print('Total Offload Queue Bits: ', offloading_bits)
            # print('Total Local Queue Bits: ', local_bits)
            # print('Number of tasks on Offload queue: ', len(self.communication_queue))
            # print('Number of tasks on local Queue: ', len(self.local_queue))
            # print('')
            #print('-------------------------------------------------------------------------------------------------------------------')

                #self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
                #self.has_transmitted_this_time_slot = True
            if len(self.task_queue)>0:
                for x in range(0,len(self.task_queue)):
                    self.task_queue.pop(0)

            if(self.allocated_offloading_ratio > 0):
                self.has_transmitted_this_time_slot = True
            #print('local qeueu length: ', len(self.local_queue))
            #print('offload qeueu length: ', len(self.communication_queue))
            #print(' ')

            local_task_identities = []
            local_task_sizes_bits = []
            local_required_cycles = []
            local_latency_requirements = []
            self.Ld_lc = 0
            self.Qd_lc = 0
            if len(self.local_queue) > 0:
                local_queue_length_bits = 0
                for task in self.local_queue:
                    local_task_identities.append(task.task_identifier)
                    local_task_sizes_bits.append(task.slot_task_size)
                    local_required_cycles.append(task.required_computation_cycles)
                    local_latency_requirements.append(task.QOS_requirement.max_allowable_latency)
                    self.Ld_lc+=task.slot_task_size
                    self.Qd_lc+=1
                    local_queue_length_bits+=task.slot_task_size

                self.local_queue_length_num_tasks = len(self.local_queue)
                self.average_local_queue_length = local_queue_length_bits
            #increment_queue_timer
            local_data = {
                'Task Identity':local_task_identities,
                'Task Size Bits':local_task_sizes_bits,
                #'Required Cycles':local_required_cycles,
                'Latency requirement':local_latency_requirements
            }

           # print('local queue arrival rate: ',local_bits*1000, ' bits/s')
            df = pd.DataFrame(data=local_data)
            #print('-----------------------------------------------------------------------------')
            # print('local queue data')
            # print(df)
            # print(' ')

            offload_task_identities = []
            offload_task_sizes_bits = []
            offload_required_cycles = []
            offload_latency_requirements = []

            if len(self.communication_queue) > 0:
                offload_queue_length_bits = 0
                for task in self.communication_queue:
                    offload_task_identities.append(task.task_identifier)
                    offload_task_sizes_bits.append(task.slot_task_size)
                    offload_required_cycles.append(task.required_computation_cycles)
                    offload_latency_requirements.append(task.QOS_requirement.max_allowable_latency)
                    offload_queue_length_bits+=task.slot_task_size

                self.offload_queue_length_num_tasks = len(self.communication_queue)
                self.average_offload_queue_length = offload_queue_length_bits

            offload_data = {
                'Task Identity':offload_task_identities,
                'Task Size Bits':offload_task_sizes_bits,
                #'Required Cycles':offload_required_cycles,
                'Latency requirement':offload_latency_requirements
            }
        #   #  print('offload queue arrival rate: ', offloading_bits*1000, ' bits/s')
            # print('-----------------------------------------------------------------------------')
            df = pd.DataFrame(data=offload_data)
            # print('offload queue data')
            # print(df)
            # print(' ')
            
        
            # print('Size of offloading queue: ',sum(offload_task_sizes_bits))
    
    def available_resource_time_code_block_fn(self,communication_channel):
        reshaped_allocated_RBs = np.array(self.allocated_RBs)
        reshaped_allocated_RBs = reshaped_allocated_RBs.squeeze()
        reshaped_allocated_RBs = reshaped_allocated_RBs.reshape(communication_channel.time_divisions_per_slot,communication_channel.num_allocate_RBs_upper_bound)
        self.available_resource_time_code_block = []
        for tb in range(1,communication_channel.time_divisions_per_slot+1):
                for rb in range(1,communication_channel.num_allocate_RBs_upper_bound+1):
                    RB_indicator = reshaped_allocated_RBs[tb-1][rb-1]
                    if RB_indicator == 1:
                        for cb in range(1,communication_channel.code_blocks_per_resource_time_block+1):
                            self.available_resource_time_code_block.append((tb,rb,cb))

        # print('embb: ', self.UE_label)
        # print('reshaped_allocated_RBs: ')
        # print(reshaped_allocated_RBs)
        #print('self.available_resource_time_code_block: ', self.available_resource_time_code_block)
        # print('')
    def count_and_make_unique_tuples(self,arr):
        """
        Counts repeating tuples in a numpy array of tuples and returns:
        - A dictionary with tuple counts
        - A list of unique tuples
        """
        # Convert the array of tuples to a NumPy array (if not already)
        arr = np.array(arr)
        
        # Use np.unique to count occurrences of each tuple
        unique_tuples, counts = np.unique(arr, axis=0, return_counts=True)
        
        # Combine tuples and their counts into a dictionary
        count_dict = {tuple(t): c for t, c in zip(unique_tuples, counts)}
        
        # Convert unique tuples back to a list of tuples
        unique_list = [tuple(t) for t in unique_tuples]
        
        return count_dict, unique_list

    def transmit_to_SBS(self, communication_channel, URLLC_users):
        self.timestep_counter+=1
        #Calculate the bandwidth achieved on each RB
        achieved_RB_channel_rates = []
        # achieved_RB_channel_rates_ = []
        #print('number of allocated RBs: ', len(self.allocate(d_RBs))
        count = 0
        self.find_puncturing_users(communication_channel,URLLC_users)
        #print('embb user: ', self.eMBB_UE_label, "puncturing urllc users: ", self.puncturing_urllc_users_)
        #print('')
        #print('self.allocated_RBs: ', self.allocated_RBs)
        self.number_of_allocated_RBs = sum(self.allocated_RBs)
        # print('embb user: ', self.UE_label, 'self.number_of_allocated_RBs: ', self.number_of_allocated_RBs)
        # print('number of clustered urllc users: ', self.num_of_clustered_urllc_users)
        # print('')
        reshaped_allocated_RBs = np.array(self.allocated_RBs)
        reshaped_allocated_RBs = reshaped_allocated_RBs.squeeze()#.reshape(1,communication_channel.time_divisions_per_slot*communication_channel.num_allocate_RBs_upper_bound)
        reshaped_allocated_RBs = reshaped_allocated_RBs.reshape(communication_channel.time_divisions_per_slot,communication_channel.num_allocate_RBs_upper_bound)
        # print('self.total_gain_')
        # print(self.total_gain_)
        # print('eMBB user: ', self.UE_label)
        # print('reshaped_allocated_RBs')
        # print(reshaped_allocated_RBs)
        # print('self.occupied_resource_time_blocks: ')
        # print(self.occupied_resource_time_blocks)
        occupied_resource_time_blocks_counts, occupied_resource_time_blocks_unique_array = self.count_and_make_unique_tuples(self.occupied_resource_time_blocks)
        # print('counts: ', counts)
        #print('occupied_resource_time_blocks_unique_array: ', occupied_resource_time_blocks_unique_array,'occupied_resource_time_blocks_counts: ', occupied_resource_time_blocks_counts)

        if self.battery_energy_level > 0:
            for tb in range(0,communication_channel.time_divisions_per_slot):
                for rb in range(0,communication_channel.num_allocate_RBs_upper_bound):
                    RB_indicator = reshaped_allocated_RBs[tb][rb]
                    current_rb_occupied = False
                    punture_counts = 0
                    for occupied_resource_time_block in occupied_resource_time_blocks_unique_array:
                        #print('occupied_resource_time_block: ', occupied_resource_time_block)
                        if occupied_resource_time_block[0] == tb+1 and occupied_resource_time_block[1] == rb+1 and occupied_resource_time_block[2] == 1:
                            current_rb_occupied = True
                            punture_counts = occupied_resource_time_blocks_counts[occupied_resource_time_block]
                            #print('occupied_resource_time_block: ', occupied_resource_time_block, 'punture_counts: ', punture_counts)
                            #break
                        elif occupied_resource_time_block[0] == tb+1 and occupied_resource_time_block[1] == rb+1 and occupied_resource_time_block[2] == 0:
                            current_rb_occupied = False
                            punture_counts = 0
                            #print('occupied_resource_time_block: ', occupied_resource_time_block, 'punture_counts: ', punture_counts)
                            #break
                    # print('tb: ', tb+1, ' rb: ', rb+1, ' currently occupied: ', current_rb_occupied)
                    # print('')
                    #if RB_indicator == 1:
                    #print(self.total_gain_)
                    #print('time_block: ', tb+1, 'resource_block: ', rb+1, 'puncture_count: ', punture_counts)
                    if len(self.total_gain_) > 1:
                        rb_small_scale_gain = self.total_gain_[rb]
                        rb_large_scale_gain = self.total_gain_[communication_channel.num_allocate_RBs_upper_bound+rb]
                    else:
                        rb_small_scale_gain = self.total_gain_[0][rb]
                        rb_large_scale_gain = self.total_gain_[0][communication_channel.num_allocate_RBs_upper_bound+rb]
                    RB_channel_gain = rb_small_scale_gain*rb_large_scale_gain
                    achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied, punture_counts)
                    #achieved_RB_channel_rate_ = self.calculate_channel_rate_(communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied)
                    achieved_RB_channel_rates.append(achieved_RB_channel_rate)
                    #achieved_RB_channel_rates_.append(achieved_RB_channel_rate_)

            self.achieved_channel_rate = sum(achieved_RB_channel_rates)
            self.embb_rate_expectation_over_prev_T_slot_(10,self.achieved_channel_rate)
            #print('achieved_channel_rate: ', self.achieved_channel_rate)
            #print('')
            #print('')
            #print('offload queue service rate: ', self.achieved_channel_rate, ' bits/s')
            #self.achieved_channel_rate_ = sum(achieved_RB_channel_rates_)
            self.previous_channel_rate = self.achieved_channel_rate
            min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
            #if self.timeslot_counter >= 500000:
            #    self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,7000],[0,10]) 
            #else:
            self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,100342857],[0,1]) 
            #self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,56000],[0,1]) 
       

    def set_channel_gain_scaling_factor(self,channel_gain_scaling_factor):
        self.channel_gain_scaling_factor = channel_gain_scaling_factor

    def calculate_channel_rate(self, communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied, punture_counts):
        RB_bandwidth = communication_channel.RB_bandwidth_Hz
        noise_spectral_density = communication_channel.noise_spectral_density_W
        channel_rate_numerator = self.assigned_transmit_power_W*RB_channel_gain#*self.channel_gain_scaling_factor
        # print('self.assigned_transmit_power_W: ', self.assigned_transmit_power_W)
        # print('RB_channel_gain: ', RB_channel_gain)
        channel_rate_denominator = noise_spectral_density*RB_bandwidth
        # print('channel_rate_numerator: ', channel_rate_numerator)
        # print('channel_rate_denominator: ', channel_rate_denominator)
        half_num_mini_slots_per_rb = communication_channel.num_of_mini_slots/2
        if current_rb_occupied == False:
            channel_rate = RB_indicator*(RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator)))
            # print('RB_indicator: ',RB_indicator)
            # print('channel_rate: ',channel_rate)
            # print('current_rb_occupied: ',current_rb_occupied)
        elif current_rb_occupied == True:
            channel_rate = RB_indicator*RB_bandwidth*(1-(punture_counts/communication_channel.num_of_mini_slots))*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
            #print('punture_counts: ', punture_counts)
            #og_channel_rate = RB_indicator*RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
        #     print('RB_indicator: ',RB_indicator)
        #     print('channel_rate: ',channel_rate)
        #     print('current_rb_occupied: ',current_rb_occupied)
        #     print('1-(1/punture_counts): ',1-(punture_counts/communication_channel.num_of_mini_slots))
        # # #return (channel_rate/500)
        return (channel_rate)
    
    # def calculate_channel_rate_(self, communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied):
    #     RB_bandwidth = communication_channel.RB_bandwidth_Hz
    #     noise_spectral_density = communication_channel.noise_spectral_density_W
    #     channel_rate_numerator = self.assigned_transmit_power_W*RB_channel_gain
    #     channel_rate_denominator = noise_spectral_density#*RB_bandwidth
    #     half_num_mini_slots_per_rb = communication_channel.num_of_mini_slots/2
    
    #     channel_rate = RB_indicator*(RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator)))
 
    #     return (channel_rate/1000)
    def local_queueing_traffic_reward(self):
        arrival_rate_tasks_per_slot = (1-self.allocated_offloading_ratio)*self.task_arrival_rate_tasks_per_second#*self.average_task_size
        service_rate_tasks_per_slot = self.max_bits_process_per_slot/self.average_task_size#len(self.dequeued_local_tasks)
        local_traffic_intensity = 0
        reward = 0
        #if len(self.dequeued_local_tasks) > 0:
        if arrival_rate_tasks_per_slot <= service_rate_tasks_per_slot:
            reward = 1
        else:
            local_traffic_intensity = arrival_rate_tasks_per_slot/service_rate_tasks_per_slot
            reward = 1-local_traffic_intensity
        # else:
        #     reward = -1

        self.local_traffic_intensity = local_traffic_intensity

        if arrival_rate_tasks_per_slot <= service_rate_tasks_per_slot:
            self.local_queueing_traffic_constraint_violation_count = 0
        else:
            self.local_queueing_traffic_constraint_violation_count = 1
        return reward
    


    def local_processing(self):
        cpu_cycles_left = self.max_service_rate_cycles_per_slot #check if 
        self.achieved_local_energy_consumption = 0
        self.dequeued_local_tasks.clear()
        used_cpu_cycles = 0
        counter = 0
        #print('self.max_bits_process_per_slot: ', self.max_bits_process_per_slot)
        #print('local queue service rate: ', (self.max_service_rate_cycles_per_slot/330)*8*1000, ' bits/s')
        #print('len(self.local_queue): ', self.local_queue)
        total_bits_size = 0
        for local_task in self.local_queue:
            total_bits_size+=local_task.slot_task_size

        self.loc_queue_length = len(self.local_queue)
        #print('total_bits_size: ', total_bits_size)
        processed_bits = 0
        for local_task in self.local_queue:
            # print('cycles left: ', cpu_cycles_left)
            # print('local_task.required_computation_cycles: ', local_task.required_computation_cycles)
            if cpu_cycles_left > local_task.required_computation_cycles:
                #print('cycles left: ', cpu_cycles_left)
                processed_bits+=local_task.slot_task_size
                #self.achieved_local_energy_consumption += self.energy_consumption_coefficient*math.pow(local_task.required_computation_cycles,2)*local_task.required_computation_cycles
                cpu_cycles_left-=local_task.required_computation_cycles
                self.dequeued_local_tasks.append(local_task)
                counter += 1

            elif cpu_cycles_left < local_task.required_computation_cycles and cpu_cycles_left > self.cycles_per_bit:
                # print('cycles left: ', cpu_cycles_left)
                # print('local_task.required_computation_cycles: ', local_task.required_computation_cycles)
                #print('task_identifier: ', local_task.task_identifier)#, 'self.bits: ', local_task.bits)
                bits_that_can_be_processed = cpu_cycles_left/self.cycles_per_bit
                processed_bits+=bits_that_can_be_processed
                cpu_cycles_left = 0
                #print("bits_that_can_be_processed: ", bits_that_can_be_processed)
                #self.achieved_local_energy_consumption += self.energy_consumption_coefficient*math.pow(cpu_cycles_left,2)*cpu_cycles_left
                #print('self.achieved_local_energy_consumption: ', self.achieved_local_energy_consumption)
                local_task.split_task(bits_that_can_be_processed) 
                break
        #print('len(self.local_queue): ', len(self.local_queue))
        for x in range(0,counter):
            self.local_queue.pop(0)
        #self.energy_consumption_coefficient*math.pow(self.max_service_rate_cycles_per_slot,2) = energy consumed per cycle (J/cycle)
        used_cpu_cycles = self.max_service_rate_cycles_per_slot - cpu_cycles_left
        # print('self.energy_consumption_coefficient: ', self.energy_consumption_coefficient)
        # print('self.max_service_rate_cycles_per_slot: ', self.max_service_rate_cycles_per_slot)
        #print('self.max_bits_process_per_slot: ', self.max_bits_process_per_slot)
        if total_bits_size > self.max_bits_process_per_slot:
            self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.max_service_rate_cycles_per_slot,2)*self.max_service_rate_cycles_per_slot#used_cpu_cycles
        else:
            self.achieved_local_energy_consumption = (total_bits_size/self.max_bits_process_per_slot) * (self.energy_consumption_coefficient*math.pow(self.max_service_rate_cycles_per_slot,2)*self.max_service_rate_cycles_per_slot)
        #print("self.achieved_local_energy_consumption: ", self.achieved_local_energy_consumption)
        # print('self.max_service_rate_cycles_per_slot: ', self.max_service_rate_cycles_per_slot)
        # print('self.achieved_local_energy_consumption: ', self.achieved_local_energy_consumption)
        #print('used_cpu_cycles: ', used_cpu_cycles)
        #self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.max_service_rate_cycles_per_slot,2)*used_cpu_cycles
        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        dequeued_task_size = []
        total_sum_size_dequeued_tasks = []
        lc_cpu_service_rate = []
        processed_bits_ = []

        self.local_queue_delay_ = 0

        if len(self.dequeued_local_tasks) > 0:
            for dequeued_local_task in self.dequeued_local_tasks:
                task_identities.append(dequeued_local_task.task_identifier)
                task_latency_requirements.append(dequeued_local_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(dequeued_local_task.queue_timer)
                dequeued_task_size.append(dequeued_local_task.slot_task_size)
                lc_cpu_service_rate
                self.local_queue_delay_+=dequeued_local_task.queue_timer

            for dequeued_local_task in self.dequeued_local_tasks:
                total_sum_size_dequeued_tasks.append(sum(dequeued_task_size))
                  

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency,
                "Size of Dequeued Task":dequeued_task_size,
                "Sum size of all Dequeued Tasks":total_sum_size_dequeued_tasks
            }

            df = pd.DataFrame(data=data)

            # print('Completed From Local Queue')
            # print('self.max_bits_process_per_slot: ', round(self.max_bits_process_per_slot))
            # print('Total processed bits this frame: ', round(processed_bits))
            # print(df)
            # print(' ')
            # print('Local Computation energy consumed in this slot: ', self.achieved_local_energy_consumption)
            #print(' ')
            #print('-----------------dequeued local tasks size total---------------------')
            #print(total_sum_size_dequeued_tasks)
            #print('')

            self.local_queue_delay_ = self.local_queue_delay_/len(self.dequeued_local_tasks)
        #average_local_queue_length
        # if self.UE_label == 1:
        #     print('self.local_queue_delay_: ', self.local_queue_delay_)
        #print(' ')
        #print(' ')

        min_local_energy_consumption, max_local_energy_consumption = self.min_and_max_achievable_local_energy_consumption()
        min_local_computation_delay, max_local_computation_delay = self.min_max_achievable_local_processing_delay()
        #print('min local delay: ', min_local_computation_delay, ' max local delay: ', max_local_computation_delay)
        #self.achieved_local_energy_consumption = interp(self.achieved_local_energy_consumption,[min_local_energy_consumption,max_local_energy_consumption],[0,5000])
        self.achieved_local_processing_delay = 1#interp(self.achieved_local_processing_delay,[min_local_computation_delay,max_local_computation_delay],[0,500])
        #print('')

    def offloading(self,communication_channel):
        offloading_bits = 0
        counter = 0
        self.dequeued_offload_tasks.clear()
        self.communication_queue_size_before_offloading = 0

        for offloading_task in self.communication_queue:
            self.communication_queue_size_before_offloading += offloading_task.slot_task_size
            self.average_packet_size_bits+=offloading_task.slot_task_size

        #print('Total bits offload_queue: ', self.average_packet_size_bits)
        #print('embb user: ', self.UE_label, 'offload queue bits: ', self.average_packet_size_bits, 'channel rate bits/slot: ', self.achieved_channel_rate/1000)

        if len(self.communication_queue) > 0:
            self.average_packet_size_bits = self.average_packet_size_bits/len(self.communication_queue)
        else:
            self.average_packet_size_bits

        # self.achieved_channel_rate 
        # print('achieved channel rate')
        #self.achieved_channel_rate = 3247758
        self.com_queue_length = len(self.communication_queue)
        if self.achieved_channel_rate == 0:
            self.achieved_transmission_delay = 0
        else:
            #left_bits = communication_channel.long_TTI*self.achieved_channel_rate
            left_bits = self.achieved_channel_rate/1000
            for offloading_task in self.communication_queue:
                if offloading_task.slot_task_size < left_bits:
                    offloading_bits += offloading_task.slot_task_size
                    left_bits -= offloading_task.slot_task_size
                    self.dequeued_offload_tasks.append(offloading_task)
                    counter+=1

                elif offloading_task.slot_task_size > left_bits:
                    offloading_task.split_task(left_bits)
                    offloading_bits+=left_bits
                    break

            for x in range(0,counter):
                self.communication_queue.pop(0)
            self.achieved_transmission_delay = 1#self.packet_offload_size_bits/self.achieved_channel_rate

        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        achieved_throughput = []
        number_of_allocated_RBs = []
        total_size_bits_offloaded = []
        task_sizes = []
        has_transmitted = False
        self.offload_queue_delay_ = 0
        if len(self.dequeued_offload_tasks) > 0:
            has_transmitted = True
            for dequeued_offload_task in self.dequeued_offload_tasks:
                task_identities.append(dequeued_offload_task.task_identifier)
                task_latency_requirements.append(dequeued_offload_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(dequeued_offload_task.queue_timer)
                achieved_throughput.append(self.achieved_channel_rate/1000)
                number_of_allocated_RBs.append(sum(self.allocated_RBs))
                task_sizes.append(dequeued_offload_task.slot_task_size)
                self.offload_queue_delay_+=dequeued_offload_task.queue_timer
                

            for dequeued_offload_task in self.dequeued_offload_tasks:    
                total_size_bits_offloaded.append(sum(task_sizes))

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency,
                "Number of allocated RBs": number_of_allocated_RBs,
                "Attained Throughput": achieved_throughput,
                "Offloaded Task Size":task_sizes,
                "Sum size of all offlaoded tasks": total_size_bits_offloaded
            }

            df = pd.DataFrame(data=data)

            # print('Completed from offloading queue')
            # print('Offloaded bits: ', offloading_bits)
            # print(df)
            # print(' ')
            # print('Achieved TTI channel rate: ', self.achieved_channel_rate/1000)
            # print(' ')
            self.offload_queue_delay_ = self.offload_queue_delay_/len(self.dequeued_offload_tasks)
        # if self.UE_label == 1:
        #     print('self.offload_queue_delay_: ', self.offload_queue_delay_)
        #     #print('')
        self.check_completed_tasks()
        #self.achieved_transmission_delay = 1
        if has_transmitted == True:
            self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*(1/communication_channel.time_divisions_per_slot)*(10**-3)*sum(self.allocated_RBs)
        #print('self.achieved_transmission_energy_consumption: ', self.achieved_transmission_energy_consumption)
        #self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*self.achieved_transmission_delay
        #print('self.achieved_transmission_energy_consumption: ', self.achieved_transmission_energy_consumption)
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[0,12*math.pow(10,-5)],[0,100])
        #print('transmission energy consumed: ', self.achieved_transmission_energy_consumption)
        #min_offload_energy_consumption, max_offload_energy_consumption = self.min_and_max_achievable_offload_energy_consumption(communication_channel)
        #min_offloading_delay, max_offloading_delay = self.min_max_achievable_offload_delay(communication_channel)
        #print('min offload delay: ', min_offloading_delay, ' max offload delay: ', max_offloading_delay)
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[min_offload_energy_consumption,max_offload_energy_consumption],[0,5000])
        #self.achieved_transmission_delay = 1#interp(self.achieved_transmission_delay,[min_offloading_delay,max_offloading_delay],[0,5000])
        #print('offload delay: ', self.achieved_transmission_delay)
        #print('transmission energy consumed: ', self.achieved_transmission_energy_consumption)
    
    def check_completed_tasks(self):
        local_queue_task_identities = []
        offload_queue_task_identities = []
        dequeued_offload_task_identities = []
        self.completed_tasks.clear()

        for dequeued_offload_task in self.dequeued_offload_tasks:
            dequeued_offload_task_identities.append(dequeued_offload_task.task_identifier)
        # First collect task identities of tasks which are currently in the local and offload queues
        for local_queue_task in self.local_queue:
            local_queue_task_identities.append(local_queue_task.task_identifier)

        for offload_queue_task in self.communication_queue:
            offload_queue_task_identities.append(offload_queue_task.task_identifier)

        # Find tasks dequeued from the local queue which are not present in the offloading queue (completed on local)
        for local_dequeued_task in self.dequeued_local_tasks:
            if local_dequeued_task.task_identifier not in offload_queue_task_identities: #and local_dequeued_task.task_identifier not in dequeued_offload_task_identities:
                self.completed_tasks.append(local_dequeued_task)

        for offload_dequeued_task in self.dequeued_offload_tasks:
            if offload_dequeued_task.task_identifier not in local_queue_task_identities:
                self.completed_tasks.append(offload_dequeued_task) 

        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        task_local_queue_latency = []
        task_offload_queue_latency = []
        if len(self.completed_tasks) > 0:
            for completed_task in self.completed_tasks:
                task_identities.append(completed_task.task_identifier)
                task_latency_requirements.append(completed_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(completed_task.queue_timer)    
                task_local_queue_latency.append(completed_task.local_queue_timer)
                task_offload_queue_latency.append(completed_task.offload_queue_timer)          

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency,
                "Local Queue Latency" : task_local_queue_latency,
                "Offload Queue Latency" : task_offload_queue_latency
            }


            df = pd.DataFrame(data=data)

            # print('Completed Tasks')
            # print(df)
            # print(' ')
            # print(' ')

        sum_latency = 0
        offload_latency = 0
        local_latency = 0
    
        for completed_task in self.completed_tasks:
            # if completed_task.QOS_requirement.max_allowable_latency < completed_task.queue_timer:
            #     sum_latency+= (completed_task.QOS_requirement.max_allowable_latency - completed_task.queue_timer)
            sum_latency+=completed_task.queue_timer
            offload_latency+=completed_task.offload_queue_timer
            local_latency+=completed_task.local_queue_timer

        self.queuing_delay = sum_latency
        #print('self.queuing_delay', self.queuing_delay)
        if len(self.completed_tasks) > 0:
            self.queuing_latency = sum_latency/len(self.completed_tasks)
            self.local_queueing_latency = local_latency/len(self.completed_tasks)
            self.offload_queueing_latency = offload_latency/len(self.completed_tasks)
        else:
            self.queuing_latency = 0
            self.local_queueing_latency = 0
            self.offload_queueing_latency = 0

        # if self.UE_label == 1:
        #     print('self.queuing_latency: ', self.queuing_latency)
        #     print('')
        #     print('self.local_queueing_latency: ', self.local_queueing_latency)
        #     print('self.offload_queueing_latency: ', self.offload_queueing_latency)
        #print('self.queuing_latency: ', self.queuing_latency)
        

    def total_energy_consumed(self):
        if self.battery_energy_level >  self.achieved_total_energy_consumption:
            #print('self.achieved_local_energy_consumption: ', self.achieved_local_energy_consumption, ' self.achieved_transmission_energy_consumption: ', self.achieved_transmission_energy_consumption)
            self.achieved_total_energy_consumption = self.achieved_local_energy_consumption + self.achieved_transmission_energy_consumption
            #print('self.achieved_total_energy_consumption: ', self.achieved_total_energy_consumption, " J")
            self.achieved_total_energy_consumption_normalized = interp(self.achieved_total_energy_consumption,[0,25],[0,1])
            #self.achieved_total_energy_consumption_normalized = interp(self.achieved_total_energy_consumption,[0,46000],[0,1])
            self.episode_energy+=self.achieved_total_energy_consumption
            #print('embb user: ', self.UE_label, "self.achieved_total_energy_consumption: ", self.achieved_total_energy_consumption)
            #print('self.achieved_total_energy_consumption: ', self.achieved_total_energy_consumption)
            self.battery_energy_level = self.battery_energy_level - self.achieved_total_energy_consumption
            if self.battery_energy_level < 0:
                self.battery_energy_level = 0
        else:
            self.achieved_total_energy_consumption = 0
            self.tasks_dropped+=1

        #print(self.battery_energy_level)
        #print('total energy: ', self.achieved_total_energy_consumption)

    def total_processing_delay(self):
        self.achieved_total_processing_delay = self.achieved_local_processing_delay + self.achieved_transmission_delay
        #print('eMBB User: ', self.eMBB_UE_label, 'achieved delay: ', self.achieved_total_processing_delay)
        #print(' ')
        #print('offload ratio: ', self.allocated_offloading_ratio, 'local delay: ', self.achieved_local_processing_delay, 'offlaod delay: ', self.achieved_transmission_delay)
        
    
    def calculate_channel_gain(self,communication_channel):
        #Pathloss gain
        #self.pathloss_gain = (math.pow(10,(35.3+37.6*math.log10(self.distance_from_SBS))))/10
        number_of_RBs = communication_channel.num_allocate_RBs_upper_bound
        small_scale_gain = np.random.exponential(1,size=(1,number_of_RBs))
        large_scale_gain = np.random.exponential(1,size=(1,number_of_RBs))

        # print('large_scale_gain: ', large_scale_gain)
        #print()
        num_samples = number_of_RBs
        #Gaussian distributed g_l with mean 0 and standard deviation 8 dB
        #print("self.timestep_counter_: ", self.timestep_counter_)
        #if self.timestep_counter_ == 1 or self.timestep_counter_ == 10:
        g_l = np.random.normal(loc=0, scale=8, size=num_samples)
        # Calculate g
        #print('************************************************************')
        # print('self.distance_from_SBS: ', self.distance_from_SBS_)
        #print('g_l: ', g_l)
        g = 35.3 + 37.8 * np.log10(self.distance_from_SBS_) + g_l
        #print('g: ', g)
        # Calculate G
        G = 10 ** (-g/10)
        G = np.array([G])
        #print('G: ', G)
        large_scale_gain = G#np.random.exponential(1,size=(1,number_of_RBs))#G
        # print('small_scale_gain: ', small_scale_gain)
        # print('large_scale_gain: ', large_scale_gain)
        # if self.timestep_counter_ == 10:
        #     self.timestep_counter_ = 0
        # #print('G: ', G)
        #print('large_scale_gain: ', large_scale_gain)
        # else:
        #     large_scale_gain = self.large_scale_gain
        # #print('************************************************************')
        #self.timestep_counter_+=1


        self.small_scale_channel_gain = small_scale_gain
        first_large_scale_gain = G#large_scale_gain[0][0]
        item = 0
        # print('large_scale_gain: ', large_scale_gain[0])
        # for gain in large_scale_gain[0]:
        #     print('gain: ', gain)
        #     print('first_large_scale_gain: ', first_large_scale_gain)
        #     large_scale_gain[0][item] = first_large_scale_gain
        #     item+=1

        average_small_scale_gain = 0
        average_large_scale_gain = 0

        average_small_scale_gain = sum(small_scale_gain[0])/len(small_scale_gain[0])

        average_large_scale_gain = sum(large_scale_gain[0])/len(large_scale_gain[0])

        self.small_scale_gain_ = average_small_scale_gain#small_scale_gain[0][0]
        self.large_scale_gain_ = average_large_scale_gain#large_scale_gain[0][0]

        self.small_scale_gain = small_scale_gain
        self.large_scale_gain = large_scale_gain
    
        # print('small_scale_gain')
        # print(self.small_scale_gain)
        # print('larger_scale_gain')
        # print(self.large_scale_gain)
        self.total_gain_ = np.concatenate((small_scale_gain,large_scale_gain),axis=1)#np.random.exponential(1,size=(1,number_of_RBs))
        # print('self.total_gain')
        # print(self.total_gain_)
        #self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        #self.total_gain = self.small_scale_channel_gain#*self.large_scale_channel_gain#self.pathloss_gain
        #if self.total_gain < 0.1:
        #    self.total_gain = 0.1

    def calculate_offloading_rate(self):
        loops = 10**3
        transmit_power = 50*math.pow(10,-3)
        RB_bandwidth = 180*math.pow(10,3)
        N_o = (math.pow(10,(-174/10)))/1000
        rates = []

        for x in range(1,loops):
            small_scale_gain = np.random.exponential(1,size=(1,1))
            g_l = np.random.normal(loc=0, scale=8, size=1)
            g = 35.3 + 37.8 * np.log10(self.distance_from_SBS_) + g_l
            large_scale_gain = 10 ** (-g/10)
            #print('large_scale_gain: ', large_scale_gain)
            channel_rate_numerator = transmit_power*small_scale_gain*large_scale_gain
            channel_rate_denominator = RB_bandwidth*N_o
            rate = RB_bandwidth*math.log2(1+channel_rate_numerator/channel_rate_denominator)
            rates.append(rate)
        
        self.average_offloading_rate = sum(rates)/(len(rates))
        #print('self.average_offloading_rate: ', self.average_offloading_rate)

    def calculate_assigned_transmit_power_W(self):
        #self.assigned_transmit_power_W = self.assigned_transmit_power_dBm#(math.pow(10,(self.assigned_transmit_power_dBm/10)))/1000
        self.assigned_transmit_power_W = (math.pow(10,(self.assigned_transmit_power_dBm/10)))/1000

    def dequeue_packet(self):
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                self.communication_queue[0].packet_queue.pop(0)

            elif len(self.communication_queue[0].packet_queue) == 0:
                self.dequeue_task()

    def dequeue_task(self):
        self.communication_queue.pop(0)

    def min_and_max_achievable_rates(self, communication_channel):
        min_num_RB = communication_channel.num_allocate_RBs_lower_bound
        max_num_RB = communication_channel.num_allocate_RBs_upper_bound
        RB_bandwidth_Hz = communication_channel.RB_bandwidth_Hz
        min_channel_rate_numerator = self.min_channel_gain*self.min_transmission_power_W
        max_channel_rate_numerator = self.max_channel_gain*self.max_transmission_power_W
        channel_rate_denominator = communication_channel.noise_spectral_density_W*RB_bandwidth_Hz
        min_achievable_rate = min_num_RB*(RB_bandwidth_Hz*math.log2(1+(min_channel_rate_numerator/channel_rate_denominator)))
        max_achievable_rate = max_num_RB*(RB_bandwidth_Hz*math.log2(1+(max_channel_rate_numerator/channel_rate_denominator)))
        return min_achievable_rate, max_achievable_rate
    
    def min_and_max_achievable_local_energy_consumption(self):
        #Local Consumption
        cycles_per_bit_max = self.cpu_cycles_per_byte*8*(5000*8000)
        achieved_local_energy_consumption_max = self.energy_consumption_coefficient*math.pow(self.cpu_clock_frequency,2)*cycles_per_bit_max
        achieved_local_energy_consumption_min = 0
        #print("max achievable local energy consumption: ",achieved_local_energy_consumption_max)
        return achieved_local_energy_consumption_min, achieved_local_energy_consumption_max
    
    def min_max_achievable_local_processing_delay(self):
        cycles_per_bit_max = self.cpu_cycles_per_byte*8*(5000*8000)
        cycles_per_bit_min = self.cpu_cycles_per_byte*8*(1000*8000)
        achieved_local_processing_delay_max = cycles_per_bit_max/self.cpu_clock_frequency
        achieved_local_processing_delay_min = cycles_per_bit_min/self.cpu_clock_frequency
        return achieved_local_processing_delay_min, achieved_local_processing_delay_max
    
    def min_and_max_achievable_offload_energy_consumption(self,communication_channel):
        #Offloading energy
        min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        max_achieved_transmission_delay = (5000*8000)/min_achievable_rate
        achieved_transmission_energy_consumption_max = self.max_transmission_power_W*max_achieved_transmission_delay
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[0,12*math.pow(10,-5)],[0,100])
        achieved_transmission_energy_consumption_min = 0
        #print("max achievable offloading energy consumption: ", achieved_transmission_energy_consumption_max)
        return achieved_transmission_energy_consumption_min, achieved_transmission_energy_consumption_max
    
    def min_max_achievable_offload_delay(self,communication_channel):
        min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        max_achieved_transmission_delay = (5000*8000)/min_achievable_rate
        min_achieved_transmission_delay = (1000*8000)/max_achievable_rate
        return min_achieved_transmission_delay, max_achieved_transmission_delay
    
    def calculate_delay_penalty(self):
        
        if (self.allowable_latency - self.achieved_total_processing_delay) >= 0:
            delay_reward = self.delay_reward
        else:
            delay_reward = (self.allowable_latency - self.achieved_total_processing_delay)
        #print('self.allowable_latency: ', self.allowable_latency)
        #print('self.achieved_total_processing_delay: ', self.achieved_total_processing_delay)
        #print('self.allowable_latency - self.achieved_total_processing_delay: ', self.allowable_latency - self.achieved_total_processing_delay)
        delay_reward = self.allowable_latency - self.achieved_total_processing_delay
        min_delay = -2200
        max_delay = 1980
        delay_reward = interp(delay_reward,[min_delay,max_delay],[0,1])
        return delay_reward
    
    def calculate_energy_efficiency(self):
        if self.achieved_total_energy_consumption == 0:
            energy_efficiency = 0
        else:
            energy_efficiency = self.achieved_channel_rate_normalized/self.achieved_total_energy_consumption_normalized#self.achieved_channel_rate#/self.achieved_total_energy_consumption #0.4*self.achieved_channel_rate_normalized/0.6*self.achieved_total_energy_consumption_normalized
            #energy_efficiency = 1/self.achieved_total_energy_consumption_normalized#self.achieved_channel_rate#/self.achieved_total_energy_consumption #0.4*self.achieved_channel_rate_normalized/0.6*self.achieved_total_energy_consumption_normalized  
            
            #energy_efficiency = self.achieved_total_energy_consumption_normalized 
            
        min_energy_efficiency = 0
        max_energy_efficiency = 10
 
        #energy_efficiency = interp(energy_efficiency,[min_energy_efficiency,max_energy_efficiency],[50000,200000])
        return energy_efficiency
    
    def calculate_resource_allocation_reward(self,communication_channel):
        offload_queue_size = self.communication_queue_size_before_offloading
        #normalize queue size
        #queue_size_normalized = interp(queue_size,[0,self.max_communication_qeueu_size],[0,1])
        #print('queue size: ', queue_size)

        #normalize achieved thoughput
        #min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        #achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[min_achievable_rate,max_achievable_rate],[0,1])
        resource_allocation_reward = self.achieved_channel_rate - offload_queue_size
        #print('throughput reward: ', throughput_reward)
        #Normailze throughput reward
        min_resource_allocation_reward = -offload_queue_size
        max_resource_allocation_reward = 7000
        #if(throughput_reward > 0):
        resource_allocation_rewardd_normalized = interp(resource_allocation_reward,[min_resource_allocation_reward,max_resource_allocation_reward],[0,5])
        #else:
            #throughput_reward_normalized = -0.65
        return resource_allocation_rewardd_normalized

    def compute_battery_energy_level(self):
        self.previous_slot_battery_energy = self.battery_energy_level
        self.battery_energy_level = self.battery_energy_level + self.energy_harvested
        if self.battery_energy_level > self.max_battery_capacity:
            self.battery_energy_level = self.max_battery_capacity
        #print('self.battery_energy_level: ', self.battery_energy_level, " J")

        self.battery_energy_level_ = self.previous_slot_battery_energy

    def harvest_energy(self):
        self.energy_harvested = np.random.exponential(250)#random.randint(0,2000)
        small_scale_gain = self.small_scale_gain[0]
        large_scale_gain = self.large_scale_gain[0]
        total_gain = sum(small_scale_gain*large_scale_gain)
        energy_harvesting_noise = np.random.uniform(-0.00001,0.00001)
        self.energy_harvested = (self.energy_conversion_efficiency*self.BS_transmit_power*self.antenna_gain*self.slot_time_ms)/(self.distance_from_SBS_**self.pathloss_coefficient)

        self.energy_harvested+=energy_harvesting_noise
        self.energy_harvested_sim = self.energy_harvested
        #print('self.energy_harvested: ', self.energy_harvested, " J")


    def energy_consumption_reward(self):
        energy_reward = self.previous_slot_battery_energy + self.energy_harversted - self.achieved_total_energy_consumption
        self.battery_energy_level_sim = self.battery_energy_level

        max_energy_reward = 40000
        min_energy_reward = -10000

        #energy_reward_normalized = 0

        if energy_reward >= 0:
            energy_reward = 1
            self.battery_energy_constraint_violation_count = 0
        else:
            energy_reward = energy_reward
            self.battery_energy_constraint_violation_count = 1

        return energy_reward
    
    def increment_task_queue_timers(self):
        if len(self.task_queue) > 0:
            for task in self.task_queue:
                task.increment_queue_timer()

        if len(self.local_queue) > 0:
            for local_task in self.local_queue:
                local_task.increment_queue_timer()
                local_task.local_queue_timer+=1

        if len(self.communication_queue) > 0:
            for offload_task in self.communication_queue:
                offload_task.increment_queue_timer()
                offload_task.offload_queue_timer+=1
                #print('emBB User: ', self.UE_label,'Task Identity: ', offload_task.task_identifier, 'offload_queue_timer: ', offload_task.offload_queue_timer)

    def queueing_delay_reward(self):
        min_queueing_delay = -3000
        max_queueing_delay = 0
        
        qeueuing_delay_reward = interp(self.queuing_delay,[min_queueing_delay,max_queueing_delay],[0,1])

        #if self.queuing_delay > 0:
        #    qeueuing_delay_reward = 1
        #else:
        #    #qeueuing_delay_reward = -1#self.queuing_delay

        return  qeueuing_delay_reward#eueuing_delay_reward
    
    def calculate_queue_lengths(self):
        #Offloading Queue Length
      
        if self.timeslot_counter == 0 or self.timeslot_counter == 1:
            self.current_queue_length_off = 0
        else:
            self.previous_offloading_ratio = self.allocated_offloading_ratio
            self.previous_arrival_rate_off = self.previous_offloading_ratio*self.previous_arrival_rate
            if (self.previous_offloading_ratio*self.previous_task_size_bits) <= 0:
                self.previous_service_rate_off = 1
            else:
                self.previous_service_rate_off = self.previous_channel_rate/(self.previous_offloading_ratio*self.previous_task_size_bits)

            if self.previous_service_rate_off <= 0:
                self.previous_traffic_intensity_off = 1
            else:
                self.previous_traffic_intensity_off = self.previous_arrival_rate_off/self.previous_service_rate_off
            
            if (1-self.previous_traffic_intensity_off) == 0:
                self.current_queue_length_off = 1
            else:
                self.current_queue_length_off = math.pow(self.previous_traffic_intensity_off,2)/(1-self.previous_traffic_intensity_off)

        #print('Current Offlaoding Queue Length: ', self.current_queue_length_off)
        #Local Queue Length
        
        if self.timeslot_counter == 0 or self.timeslot_counter == 1:
            self.current_queue_length_lc = 0
        else:
            self.previous_offloading_ratio = self.allocated_offloading_ratio
            self.previous_arrival_rate_lc = (1-self.previous_offloading_ratio)*self.previous_arrival_rate
            if ((1-self.previous_offloading_ratio)*self.previous_task_size_bits) == 0:
                self.previous_service_rate_lc = 1
            else:
                self.previous_service_rate_lc = self.service_rate_bits_per_slot/((1-self.previous_offloading_ratio)*self.previous_task_size_bits)

            if self.previous_service_rate_lc <= 0:
                self.previous_traffic_intensity_lc = 1
            else:
                self.previous_traffic_intensity_lc = self.previous_arrival_rate_lc/self.previous_service_rate_lc
            
            if (1-self.previous_traffic_intensity_lc) == 0:
                self.current_queue_length_lc = 1
            else:    
                self.current_queue_length_lc = math.pow(self.previous_traffic_intensity_lc,2)/(2*(1-self.previous_traffic_intensity_lc))

        self.current_queue_length_modified_off = len(self.communication_queue)
        self.current_queue_length_modified_lc = len(self.local_queue)

        return self.current_queue_length_modified_off,self.current_queue_length_modified_lc
        #print('Current Local Queue Length: ', self.current_queue_length_lc)

    def calculate_queuing_delays(self):
        #Offload queuing delay
        #offload_queuing_delay = 1
        #local_queuing_delay = 1
        if self.current_queue_length_off < 0:
            current_arrival_rate_off = self.current_arrival_rate*self.allocated_offloading_ratio
            offload_queuing_delay = 1#-(self.current_queue_length_off/current_arrival_rate_off)
        else:
            offload_queuing_delay = 1  

        if self.current_queue_length_lc < 0:
            current_arrival_rate_lc = self.current_arrival_rate*(1-self.allocated_offloading_ratio)
            local_queuing_delay = 1#-(self.current_queue_length_lc/current_arrival_rate_lc)
        else:
            local_queuing_delay = 1

        #################################################################################################################
        offload_queuing_delay_modified = 1
        local_queuing_delay_modified = 1
        
        if self.current_queue_length_modified_off > 0:
            current_arrival_rate_off = self.current_arrival_rate*self.allocated_offloading_ratio
            if current_arrival_rate_off > 0:
                offload_queuing_delay_modified = self.current_queue_length_modified_off/current_arrival_rate_off
    
        if self.current_queue_length_modified_lc > 0:
            current_arrival_rate_lc = self.current_arrival_rate*(1-self.allocated_offloading_ratio)
            if current_arrival_rate_lc > 0:
                local_queuing_delay_modified = self.current_queue_length_modified_lc/current_arrival_rate_lc
            else:
                local_queuing_delay_modified = 999999
            

        #delay_reward = 1/max(offload_queuing_delay,local_queuing_delay)
        delay = max(offload_queuing_delay_modified,local_queuing_delay_modified)
        delay_reward = 1/delay
        max_delay_reward = 5
        min_delay_reward = 0
        delay_reward_normalized = interp(delay_reward,[min_delay_reward,max_delay_reward],[0,1])
        #self.current_queue_length_modified_off,self.current_queue_length_modified_lc
        return delay_reward,delay
    
    def find_puncturing_users(self,communication_channel,URLLC_users):
        reshaped_allocated_RBs = np.array(self.allocated_RBs)
        reshaped_allocated_RBs = reshaped_allocated_RBs.squeeze()#.reshape(1,communication_channel.time_divisions_per_slot*communication_channel.num_allocate_RBs_upper_bound)
        reshaped_allocated_RBs = reshaped_allocated_RBs.reshape(communication_channel.time_divisions_per_slot,communication_channel.num_allocate_RBs_upper_bound)
        # print("reshaped_allocated_RBs")
        # print(reshaped_allocated_RBs)
        sum_matrix = np.sum(reshaped_allocated_RBs,axis=0)
        # print('sum_matrix')
        # print(sum_matrix)
        # print('')
        r = 1
        self.allocated_resource_blocks_numbered.clear()
        for matrix in sum_matrix:
            
            if matrix == 1 or matrix == 2:
                self.allocated_resource_blocks_numbered.append(r)

            r+=1
        #print('self.allocated_resource_blocks_numbered: ', self.allocated_resource_blocks_numbered)
        r = 0
        c = 0
        binary_indicator = 0
        self.time_allocators = []
        self.time_matrix = []
        for col in range(0,communication_channel.num_allocate_RBs_upper_bound):
            self.time_allocators.clear()
            for row in range(0,communication_channel.time_divisions_per_slot):
                binary_indicator = reshaped_allocated_RBs[row][col]
                #print('row: ', row, 'col: ', col, 'binary indicator: ', binary_indicator)
                if binary_indicator == 1 and row == 0:
                    self.time_allocators.append(1)
                    #print('append 1')
                elif binary_indicator == 1 and row == 1:
                    self.time_allocators.append(2)
                    #print('append 2')

            #print('self.time_allocators: ', self.time_allocators)
            if len(self.time_allocators) == 1:
                self.time_matrix.append((self.time_allocators[0]))
            elif len(self.time_allocators) == 2:
                self.time_matrix.append((self.time_allocators[0],self.time_allocators[1]))
            elif len(self.time_allocators) == 0:
                self.time_matrix.append((0))
        # print('self.time_matrix')
        # print(self.time_matrix)
        # print('')
        self.puncturing_urllc_users(URLLC_users)

    def puncturing_urllc_users(self,urllc_users):
        self.puncturing_urllc_users_.clear()
        self.occupied_resource_time_blocks.clear()
        #self.numbers_of_puncturing_users = 0
        #print('embb: ', self.UE_label, 'self.allocated_resource_blocks_numbered: ', self.allocated_resource_blocks_numbered)
        for allocated_resource_block in self.allocated_resource_blocks_numbered:
            
            time_blocks_at_this_rb = self.time_matrix[allocated_resource_block-1]
            #print('embb: ', self.UE_label, 'time_blocks_at_this_rb: ', time_blocks_at_this_rb)
            if time_blocks_at_this_rb == 1 or time_blocks_at_this_rb == 2:
                for urllc_user in urllc_users:
                    if urllc_user.assigned_resource_block == allocated_resource_block and urllc_user.assigned_time_block == time_blocks_at_this_rb:
                        #print('urllc_user.assigned_resource_block: ', urllc_user.assigned_resource_block, 'urllc_user.assigned_time_block: ', urllc_user.assigned_time_block)
                        self.puncturing_urllc_users_.append(urllc_user.URLLC_UE_label)
                        #print('self.puncturing_urllc_users_ ***: ', self.puncturing_urllc_users_)
                        #print('urllc_user.has_transmitted_this_time_slot: ', urllc_user.has_transmitted_this_time_slot)
                        if urllc_user.has_transmitted_this_time_slot == True:
                            self.occupied_resource_time_blocks.append((time_blocks_at_this_rb,allocated_resource_block,1))
                        elif urllc_user.has_transmitted_this_time_slot == False:
                            self.occupied_resource_time_blocks.append((time_blocks_at_this_rb,allocated_resource_block,0))

            
            elif time_blocks_at_this_rb == (1,2):
                for time_block_at_this_rb in time_blocks_at_this_rb:
                    for urllc_user in urllc_users:
                        if urllc_user.assigned_resource_block == allocated_resource_block and urllc_user.assigned_time_block == time_block_at_this_rb:
                            #print('urllc_user.assigned_resource_block: ', urllc_user.assigned_resource_block, 'urllc_user.assigned_time_block: ', urllc_user.assigned_time_block)
                            self.puncturing_urllc_users_.append(urllc_user.URLLC_UE_label)
                            #print('urllc_user.has_transmitted_this_time_slot: ', urllc_user.has_transmitted_this_time_slot)
                            if urllc_user.has_transmitted_this_time_slot == True:
                                self.occupied_resource_time_blocks.append((time_block_at_this_rb,allocated_resource_block,1))
                            elif urllc_user.has_transmitted_this_time_slot == False:
                                self.occupied_resource_time_blocks.append((time_block_at_this_rb,allocated_resource_block,0))

        self.numbers_of_puncturing_users = len(self.puncturing_urllc_users_)
        #print('self.numbers_of_puncturing_users: ', self.numbers_of_puncturing_users)
        # print('channel rate: ', self.achieved_channel_rate)
        # print('')
        # print('occupied_resource_time_blocks')
        # print(self.occupied_resource_time_blocks)
        #print('embb user id: ', self.eMBB_UE_label, 'allocated rb: ', self.allocated_resource_blocks_numbered)
        # print('allocated time blocks: ', self.time_matrix)
        # print('')
        #print('self.puncturing_urllc_users_: ', self.puncturing_urllc_users_, 'self.occupied_resource_time_blocks: ', self.occupied_resource_time_blocks)
        # print('occupied resource blocks: ', self.occupied_resource_time_blocks)
        #print('')
        #print('embb: ', self.UE_label, 'Puncturing URLLC users: ', self.puncturing_urllc_users_)

        # for urllc_user in urllc_users:
        #     print('urllc_user id: ', urllc_user.URLLC_UE_label)
        #     print('allocated rb: ', urllc_user.assigned_resource_block)
        #     print('allocated time block: ', urllc_user.assigned_time_block)
        #     print('has user transmitted: ', urllc_user.has_transmitted_this_time_slot)
        #     print('')

            
            

          
            
            

        #print('')
        #
        # for URLLC_user in URLLC_users:
        #     print('URLLC users allocated RB: ', URLLC_user.assigned_resource_block)
       

     


    #def calculate_queuing_time(self):
    def urllc_puncturing_users_sum_data_rates(self, urllc_users):
        sum_data_rate = 0
        self.puncturing_users_sum_data_rates = 0
        self.num_puncturing_users = 0

        for urllc_user in urllc_users:
            if urllc_user.URLLC_UE_label in self.puncturing_urllc_users_:
                self.puncturing_users_sum_data_rates+=(urllc_user.achieved_channel_rate_per_slot*1000)
                self.num_puncturing_users+=1

    
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

    def new_time_delay_calculation(self):
        average_task_cycles_per_packet = 0
        self.computation_time_per_bit
        queue_length_bits = 0
        if len(self.local_queue) > 0:
            for task in self.local_queue:
                average_task_cycles_per_packet+=task.required_computation_cycles
                queue_length_bits+=task.slot_task_size
            average_task_cycles_per_packet = average_task_cycles_per_packet/len(self.local_queue)

        local_computation_time = average_task_cycles_per_packet/self.max_service_rate_cycles_per_slot
        local_queueing_time = queue_length_bits*self.computation_time_per_bit#len(self.local_queue)*local_computation_time
        local_delay = local_computation_time+local_queueing_time

        average_packet_size_bits = 0
        total_packet_size_bits = 0
        #print(len(self.communication_queue))
        if len(self.communication_queue) > 0:
            for task in self.communication_queue:
                total_packet_size_bits+=task.slot_task_size

            average_packet_size_bits =  total_packet_size_bits/len(self.communication_queue)

        self.average_task_size_offload_queue = average_packet_size_bits
        expected_rate_over_prev_T_slot = self.embb_rate_expectation_over_prev_T_slot(10,self.achieved_channel_rate)
        expected_rate_over_prev_T_slot_ms = expected_rate_over_prev_T_slot/1000
        self.expected_rate_over_prev_T_slot = expected_rate_over_prev_T_slot_ms
        
        #print('expected_rate_over_prev_T_slot_ms: ', expected_rate_over_prev_T_slot_ms)
        #print('average_packet_size_bits: ', average_packet_size_bits)
        #print('len(self.communication_queue): ', len(self.communication_queue))
        #print('expected_rate_over_prev_T_slot_ms: ', expected_rate_over_prev_T_slot_ms)
        #if expected_rate_over_prev_T_slot_ms > 0:
        if expected_rate_over_prev_T_slot_ms == 0:
            offload_queueing_time = self.offload_queueing_latency
        else:
            offload_queueing_time = (average_packet_size_bits/expected_rate_over_prev_T_slot_ms)*len(self.communication_queue)
        #print('eMBB: ', self.UE_label, 'offload_queueing_time: ', offload_queueing_time)
        #else:
            #offload_queueing_time = (average_packet_size_bits)*len(self.communication_queue)
        offloading_delay = offload_queueing_time + 1

        #print('offload_queueing_time: ', offload_queueing_time)

        local_queue_size_bits = 0
        offload_queue_size_bits = 0
        if len(self.local_queue) > 0:
            for task in self.local_queue:
                local_queue_size_bits+=task.slot_task_size
        
        if len(self.communication_queue) > 0:
            for task in self.communication_queue:
                offload_queue_size_bits+=task.slot_task_size

        self.local_queue_length = local_queue_size_bits
        self.offload_queue_length = offload_queue_size_bits

        # self.local_queue_length_num_tasks = len(self.local_queue)
        # self.offload_queue_length_num_tasks = len(self.communication_queue)
        # # print('self.local_queue_length: ', self.local_queue_length)
        # # print('self.offload_queue_length: ', self.offload_queue_length)
        # # print('local_delay: ', local_delay)
        # # print('offloading_delay: ', offloading_delay)
        # #print('eMBB UE: ', self.UE_label, 'local_delay: ', local_delay, 'offloading_delay: ', offloading_delay)

        # # self.local_queue_delay = local_delay
        # # self.offload_queue_delay = offloading_delay

        # # self.local_queue_lengths.append(len(self.local_queue))
        # # self.offload_queue_lengths.append(len(self.communication_queue))

        # # self.local_queue_delays.append(local_delay)
        # # self.offload_queue_delays.append(offloading_delay)
        # #self.average_local_queue_length, self.average_offload_queue_length, self.average_local_delays, self.average_offload_delays = self.avg_queue_length_delays_over_T_slots(5, self.local_queue_length, self.offload_queue_length, local_delay, offloading_delay)

        # self.average_local_queue_length = self.local_queue_length 
        # self.average_offload_queue_length = self.offload_queue_length
        self.average_local_delays = local_delay
        self.average_offload_delays = offloading_delay
        # print('local_delay: ', local_delay)
        # print('local queue length: ', self.local_queue_length)
        # print('offloading_delay: ', offloading_delay)
        # print('offload queue length: ', self.offload_queue_length)
        max_delay = max(local_delay,offloading_delay)
        max_delay_normalized =  interp(max_delay,[0,4],[0,20])
        return max_delay, max_delay_normalized

    def embb_rate_expectation_over_prev_T_slot(self, T, embb_total_rate):
        number_of_previous_time_slots = T

        if len(self.previous_rates) == number_of_previous_time_slots:
            self.previous_rates[int(self.ptr)] = embb_total_rate
            self.ptr = (self.ptr + 1) % number_of_previous_time_slots
        else:
            self.previous_rates.append(embb_total_rate)

        average_rate = sum(self.previous_rates)/len(self.previous_rates)
        return average_rate
    
    def offloading_queue_stability_constraint_reward(self):
        # offload_traffic = 0
        # self.offload_stability_constraint_reward = 0
        # #if self.achieved_channel_rate > 0:
        # #offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_packet_size_bits)/self.achieved_channel_rate
        # reward = 0
        # #average_rate = self.embb_rate_expectation_over_prev_T_slot_(10,self.achieved_channel_rate)/1000

        # # if self.UE_label == 1:
        # #     print('average_rate: ',average_rate)
        # offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_task_size)/(self.average_data_rate/1000)
        # if self.average_data_rate > 0:
        #     #if (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_task_size) <= self.average_data_rate:#self.achieved_channel_rate/1000:
        #     if (self.allocated_offloading_ratio*self.average_task_arrival_rate*self.average_task_size) <= self.average_data_rate:
        #         reward = 1
        #     else:
        #         #offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_task_size)/self.average_data_rate#(self.achieved_channel_rate/1000)
        #         offload_traffic = (self.allocated_offloading_ratio*self.average_task_arrival_rate*self.average_task_size)/self.average_data_rate
        #         reward = 1-offload_traffic
        # else:
        #     reward = -1

        # # if reward < 0:
        # #     reward = reward
        # #     #self.offload_queueing_traffic_constaint_violation_count = 1
        # # else:
        # #     reward = 1
        # #     #self.offload_queueing_traffic_constaint_violation_count = 0

        # self.offlaod_traffic_numerator = self.allocated_offloading_ratio*self.average_task_arrival_rate*self.average_task_size
        # self.offload_stability_constraint_reward = reward

        # #if (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_packet_size_bits) >= self.achieved_channel_rate/1000:
        # if (self.allocated_offloading_ratio*self.average_task_arrival_rate*self.average_task_size) >= self.average_data_rate/1000:
        #     self.offload_queueing_traffic_constaint_violation_count = 1
        # else:
        #     self.offload_queueing_traffic_constaint_violation_count = 0
        # return reward#offload_traffic
        #return reward#offload_traffic
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        offload_traffic = 0
        self.offload_stability_constraint_reward = 0
        #if self.achieved_channel_rate > 0:
        #offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_packet_size_bits)/self.achieved_channel_rate
        reward = 0
        average_rate = self.embb_rate_expectation_over_prev_T_slot_(10,self.achieved_channel_rate)/1000

        # if self.UE_label == 1:
        #     print('average_rate: ',average_rate)
        offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_task_size)/(self.average_offloading_rate/1000)
        if average_rate > 0:
            if (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_task_size) <= average_rate:#self.achieved_channel_rate/1000:
                reward = 1
            else:
                offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_task_size)/average_rate#(self.achieved_channel_rate/1000)
                reward = 1-offload_traffic
        else:
            reward = -1

        # if reward < 0:
        #     reward = reward
        #     #self.offload_queueing_traffic_constaint_violation_count = 1
        # else:
        #     reward = 1
        #     #self.offload_queueing_traffic_constaint_violation_count = 0

        self.offlaod_traffic_numerator = self.allocated_offloading_ratio*self.task_arrival_rate*self.average_packet_size_bits
        self.offload_stability_constraint_reward = reward

        if (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_packet_size_bits) >= self.achieved_channel_rate/1000:
            self.offload_queueing_traffic_constaint_violation_count = 1
        else:
            self.offload_queueing_traffic_constaint_violation_count = 0
        return reward#offload_traffic
    
    def local_queue_violation_constraint_reward(self):
        average_packet_size_bits = 0
        if len(self.local_queue) > 0:
            for task in self.local_queue:
                average_packet_size_bits+=task.slot_task_size
            average_packet_size_bits =  average_packet_size_bits/len(self.local_queue)

        if average_packet_size_bits > 0:
            average_service_rate = self.service_rate_bits_per_slot/average_packet_size_bits
        else:
            average_service_rate = 0

        if average_service_rate > 0 :
            G = ((1-self.allocated_offloading_ratio)*self.task_arrival_rate)/average_service_rate
        else:
            G = 0

        queue_length = len(self.local_queue)
        #print('queue_length: ', queue_length)
        sum_violation_probability = 0
        for i in range(0,queue_length+1):
            sum_violation_probability+=self.probabitlity_of_num_packet(i,G)

        sum_violation_probability = 1 - sum_violation_probability
        max_sum_violation_probability = 0.5
        min_sum_violation_probability = -12
        #local_queue_delay_violation_probability
        sum_violation_probability_norm = interp(sum_violation_probability,[min_sum_violation_probability,max_sum_violation_probability],[0,1])
        violation_reward = self.max_lc_queue_delay_violation_probability - sum_violation_probability_norm
        return violation_reward#sum_violation_probability_norm

    def probabitlity_of_num_packet(self,i,G):
        p1 = 0.5
        sum_second_term = 0
        for k in range(2,(i+1)+1):
            sum_second_term+= (p1*(math.pow(G,(i+k+1)))/math.factorial((i+k+1)))
            #print(sum_second_term)

        second_term = math.exp(G)*sum_second_term
        first_term = math.exp(G)*(math.pow(G,i)/math.factorial(i))*p1 

        pi = first_term+second_term

        return pi
    

    def avg_queue_length_delays_over_T_slots(self, T, local_queue_length, offload_queue_length, local_delay, offload_delay):
        self.timeslot_counter+=1
        number_of_previous_time_slots = T

        if len(self.local_queue_lengths) == number_of_previous_time_slots:
            self.local_queue_lengths[int(self.ptr)] = local_queue_length
            self.ptr_local_queue_lengths = (self.ptr_local_queue_lengths + 1) % number_of_previous_time_slots
        else:
            self.local_queue_lengths.append(local_queue_length)

        average_local_queue_length = sum(self.local_queue_lengths)/len(self.local_queue_lengths)

        ################################################################################################################
        if len(self.offload_queue_lengths) == number_of_previous_time_slots:
            self.offload_queue_lengths[int(self.ptr)] = offload_queue_length
            self.ptr_offload_queue_length = (self.ptr_offload_queue_length + 1) % number_of_previous_time_slots
        else:
            self.offload_queue_lengths.append(offload_queue_length)

        average_offload_queue_length = sum(self.offload_queue_lengths)/len(self.offload_queue_lengths)

        ################################################################################################################
        if len(self.local_delays) == number_of_previous_time_slots:
            self.local_delays[int(self.ptr)] = local_delay
            self.ptr_local_delay = (self.ptr_local_delay + 1) % number_of_previous_time_slots
        else:
            self.local_delays.append(local_delay)

        average_local_delays = sum(self.local_delays)/len(self.local_delays)

        ################################################################################################################
        if len(self.offload_delays) == number_of_previous_time_slots:
            self.offload_delays[int(self.ptr)] = offload_delay
            self.ptr_offload_delay = (self.ptr_offload_delay + 1) % number_of_previous_time_slots
        else:
            self.offload_delays.append(offload_delay)

        average_offload_delays = sum(self.offload_delays)/len(self.offload_delays)


        return average_local_queue_length, average_offload_queue_length, average_local_delays, average_offload_delays

    def local_queue_delay_violation_probability(self):        
        T_max_lc = 10*(self.average_task_size*self.computation_time_per_bit)
        #T_max_lc = 2*((1-self.allocated_offloading_ratio)*self.average_bits_tasks_arriving*self.computation_time_per_bit)
        Ld_max = round(T_max_lc/self.computation_time_per_bit)

        Ld_max = 30
        #Ld_max = 1000

        # print('self.average_bits_tasks_arriving: ', self.average_bits_tasks_arriving)
        # print('self.computation_time_per_bit: ', self.computation_time_per_bit)
        # print('T_max_lc: ', T_max_lc)
        # print('Ld_max: ', Ld_max)
        mew = self.max_service_rate_cycles_per_slot/self.cycles_per_bit
        #self.allocated_offloading_ratio = 0.5
        rho = (self.average_bits_tasks_arriving*(1-self.allocated_offloading_ratio)*self.task_arrival_rate_tasks_per_second)/mew
        #print('rho: ', rho)
        Pr_Lds = []
        queueing_violation_prob = 0
        
        # if self.Qd_lc > 0:
        for Ld_lc in range(0,self.Ld_max+1):
            Pr_Ld = self.Pr_Ld_lc(Ld_lc)
            Pr_Lds.append(Pr_Ld)
        #print('len(Pr_Lds): ', Pr_Lds)
        sum_Pr_Lds = sum(Pr_Lds)

        queueing_violation_prob = 1 - sum_Pr_Lds
        self.local_queue_delay_violation_probability_ = queueing_violation_prob
        # if self.UE_label == 1:
        #     print('local queueing_violation_prob: ',queueing_violation_prob,'sum(sum_Pr_Lds): ', sum_Pr_Lds)
        #     #print('---------------------------------------------------------------------------------')
        #     print('')
        if queueing_violation_prob > 1:
            queueing_violation_prob = 1
        elif queueing_violation_prob < 0:
            queueing_violation_prob = 0

        # print('local queueing_violation_prob: ', queueing_violation_prob)
        # print('---------------------------------------------------------------------')
        queueing_violation_prob_reward = 0
        if (self.local_queue_delay_violation_probability_constraint-(queueing_violation_prob)) < 0:
            queueing_violation_prob_reward = (self.local_queue_delay_violation_probability_constraint-(queueing_violation_prob))
            self.local_time_delay_violation_prob_constraint_violation_count = 1
        else:
            queueing_violation_prob_reward = 1
            self.local_time_delay_violation_prob_constraint_violation_count = 0
        self.local_queueing_violation_prob_reward = queueing_violation_prob_reward
        return queueing_violation_prob_reward



    def Pr_Ld_lc(self,Ld):
        Q_max = 10
        Pr_Ld_Qs = []
        for Q in range(0,Q_max+1):
            # if self.UE_label == 1 and queue_type == 'offload':
            #     print('Q: ', Q)
            #     print('Ld: ', Ld)
            #     print('Pr(Ld|Qd):',self.neg_binom_dist(Ld,Q))
            #     print('Pr(Q): ',self.Pr_Qd(Q))
            #     print('*******************************************')
            Pr_Ld_Q = self.neg_binom_dist(Ld,Q)*self.Pr_Qd_lc(Q)
            Pr_Ld_Qs.append(Pr_Ld_Q)

        Pr_Ld_Qs = np.array(Pr_Ld_Qs)
        Pr_Ld_Qs = Pr_Ld_Qs[~np.isnan(Pr_Ld_Qs)]
        Pr_Ld = sum(Pr_Ld_Qs)

        return Pr_Ld
    
    def Pr_Ld_off(self,Ld):
        Q_max = 10
        Pr_Ld_Qs = []
        for Q in range(0,Q_max+1):
            #if self.UE_label == 1:
            #     print('Q: ', Q)
            #     print('Ld: ', Ld)
                #print('Pr(Ld|Qd):',self.neg_binom_dist(Ld,Q))
            #     print('Pr(Q): ',self.Pr_Qd(Q,queue_type))
            #     print('*******************************************')
            Pr_Ld_Q = self.neg_binom_dist(Ld,Q)*self.Pr_Qd_off(Q)
            Pr_Ld_Qs.append(Pr_Ld_Q)

        Pr_Ld_Qs = np.array(Pr_Ld_Qs)
        #Pr_Ld_Qs = Pr_Ld_Qs[~np.isnan(Pr_Ld_Qs)]
        Pr_Ld = sum(Pr_Ld_Qs)

        return Pr_Ld
        
    
    def neg_binom_dist(self,Ld,Q):
        n = Ld - 1
        k = Q - 1
        #print('Ld: ', Ld, 'Q: ', Q, 'n-k: ', (n-k))
        if Q == 0 and Ld == 0:
            Pr_Ld_Qd = 1
        elif Q == 0 and Ld > 0:
            Pr_Ld_Qd = 0
        else:
            p = self.geometric_probability
            Pr_Ld_Qd = nbinom.pmf(Ld - Q, Q, p)
        #Pr_Ld_Qd = nbinom.pmf(Ld, Q, p)
        #print('Ld: ',Ld)
        #print('Pr(Ld|Qd): ', Pr_Ld_Qd)
        #print('****************************************************')
        #if n >= k:
            #Pr_Ld_Qd = (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*(p**Q)*(1-p)**(Ld-Q)
            #print('Pr_Ld_Qd 1: ', Pr_Ld_Qd)
            #print('Pr_Ld_Qd 2: ', Pr_Ld_Qd)
        #else:
        #    Pr_Ld_Qd = 0

        
        return Pr_Ld_Qd
    
    def Pr_Qd_lc(self,Q):

       
        mew = self.max_service_rate_cycles_per_slot/self.cycles_per_bit
        #print('self.allocated_offloading_ratio: ', self.allocated_offloading_ratio)
        #self.allocated_offloading_ratio =0.8
        rho = (self.average_task_size*(1-self.allocated_offloading_ratio)*self.average_task_arrival_rate)/mew
        #print('self.allocated_offloading_ratio: ',self.allocated_offloading_ratio)
        # print('self.average_bits_tasks_arriving: ', self.average_bits_tasks_arriving)
        # print('self.task_arrival_rate_tasks_per_second: ', self.task_arrival_rate_tasks_per_second)
        #print('rho: ', rho)
        #rho = 0.8

        if rho >= 1:
            rho = 0.99
        else:
            rho = rho

        #print('rho: ', rho)
        Pr_Q = (rho**Q)*(1-rho)
        #self.Pr_Qs.clear()
        return Pr_Q
    
    def embb_rate_expectation_over_prev_T_slot_(self, T, embb_total_rate):
        number_of_previous_time_slots = T

        if len(self.previous_rates_) == number_of_previous_time_slots:
            self.previous_rates_[int(self.pointer_)] = embb_total_rate
            self.pointer_ = (self.pointer_ + 1) % number_of_previous_time_slots
        else:
            self.previous_rates_.append(embb_total_rate)

        average_rate = sum(self.previous_rates_)/len(self.previous_rates_)
        self.average_data_rate = average_rate
        return average_rate
    
    def Pr_Qd_off(self,Q):
        mew = self.average_data_rate/1000#self.embb_rate_expectation_over_prev_T_slot_(10,self.achieved_channel_rate)/1000
        #print('self.allocated_offloading_ratio: ', self.allocated_offloading_ratio)
        #self.allocated_offloading_ratio =0.8
        if mew > 0:
            rho = (self.average_task_size*(self.allocated_offloading_ratio)*self.average_task_arrival_rate)/mew
        else:
            rho = 0.99


        if rho >= 1:
            rho = 0.99
        else:
            rho = rho
        
        Pr_Q = (rho**Q)*(1-rho)
        return Pr_Q
    
    def offload_ratio_reward(self):
        offload_ratio_min = 0.4
        offload_ratio_reward = 0
        if self.allocated_offloading_ratio < offload_ratio_min:
            offload_ratio_reward = offload_ratio_min - self.allocated_offloading_ratio
        else:
            offload_ratio_reward = 1

        self.offloa_ratio_reward = offload_ratio_reward
        return offload_ratio_reward
    
    def offload_queue_delay_violation_probability(self):
        T_max_lc = 10*(self.average_task_size*self.computation_time_per_bit)
        #T_max_lc = 2*((1-self.allocated_offloading_ratio)*self.average_bits_tasks_arriving*self.computation_time_per_bit)
        Ld_max = round(T_max_lc/self.computation_time_per_bit)
        Ld_max = 30

        # print('self.average_bits_tasks_arriving: ', self.average_bits_tasks_arriving)
        # print('self.computation_time_per_bit: ', self.computation_time_per_bit)
        # print('T_max_lc: ', T_max_lc)
        # print('Ld_max: ', Ld_max)
        #self.allocated_offloading_ratio = 0.5
        mew = self.average_data_rate/1000#self.embb_rate_expectation_over_prev_T_slot_(10,self.achieved_channel_rate)/1000
            #print('self.allocated_offloading_ratio: ', self.allocated_offloading_ratio)
            #self.allocated_offloading_ratio =0.8
        if mew > 0:
            rho = (self.average_task_size*(self.allocated_offloading_ratio)*self.average_task_arrival_rate)/mew
        else:
            rho = 0.99

        original_rho = rho
        if rho >= 1:
            rho = 0.99
        else:
            rho = rho

        self.rho = rho

        # if self.UE_label == 1:
        #     print('embb user: ', self.UE_label)
        #     print('mew: ', mew)
        #     print('original_rho: ',original_rho)
        #     print('rho: ', rho)
        #     print('data rates:',self.previous_rates_)
        #     print('Slot data rate: ', self.achieved_channel_rate/1000)
        Pr_Lds = []
        queueing_violation_prob = 0
        
        # if self.Qd_lc > 0:
        for Ld_off in range(0,self.Ld_max+1):
            Pr_Ld = self.Pr_Ld_off(Ld_off)
            Pr_Lds.append(Pr_Ld)
        #print('len(Pr_Lds): ', Pr_Lds)
        sum_Pr_Lds = sum(Pr_Lds)
        if sum_Pr_Lds >= 1:
            sum_Pr_Lds = 0.999876
        # print('self.Pr_Qs:')
        # print(self.Pr_Qs)

        queueing_violation_prob = 1 - sum_Pr_Lds

        if self.rho == 0.99:
            queueing_violation_prob = 0.9419094427221881
            sum_Pr_Lds = 1-queueing_violation_prob

        self.offload_queue_delay_violation_probability_ = queueing_violation_prob
        # if self.UE_label == 1:
        #     print('offload queueing_violation_prob: ',queueing_violation_prob,'sum(sum_Pr_Lds): ', sum_Pr_Lds)
        #     print('---------------------------------------------------------------------------------')
        if queueing_violation_prob > 1:
            queueing_violation_prob = 1
        elif queueing_violation_prob < 0:
            queueing_violation_prob = 0

        #print('local queueing_violation_prob: ', queueing_violation_prob)
        #print('---------------------------------------------------------------------')
        queueing_violation_prob_reward = 0
        if (self.local_queue_delay_violation_probability_constraint-(queueing_violation_prob)) < 0:
            queueing_violation_prob_reward = (self.local_queue_delay_violation_probability_constraint-(queueing_violation_prob))
            self.offload_time_delay_violation_prob_constraint_violation_count = 1
        else:
            queueing_violation_prob_reward = 1
            self.offload_time_delay_violation_prob_constraint_violation_count = 0
        self.offload_queueing_violation_prob_reward = queueing_violation_prob_reward
        return queueing_violation_prob_reward
    
    def offload_queue_delay_violation_probability__(self):
        T_max_lc = 10*(self.average_task_size*self.computation_time_per_bit)
        #T_max_lc = 2*((1-self.allocated_offloading_ratio)*self.average_bits_tasks_arriving*self.computation_time_per_bit)
        Ld_max = round(T_max_lc/self.computation_time_per_bit)
        Ld_max = 1000

        # print('self.average_bits_tasks_arriving: ', self.average_bits_tasks_arriving)
        # print('self.computation_time_per_bit: ', self.computation_time_per_bit)
        # print('T_max_lc: ', T_max_lc)
        # print('Ld_max: ', Ld_max)
        #self.allocated_offloading_ratio = 0.8
        mew = self.average_data_rate/1000#self.embb_rate_expectation_over_prev_T_slot_(10,self.achieved_channel_rate)/1000
            #print('self.allocated_offloading_ratio: ', self.allocated_offloading_ratio)
            #self.allocated_offloading_ratio =0.8
        if mew > 0:
            rho = (self.average_task_size*(self.allocated_offloading_ratio)*self.average_task_arrival_rate)/mew
        else:
            rho = 0.99

        original_rho = rho
        if rho >= 1:
            rho = 0.99
        else:
            rho = rho

        self.rho = rho

        if self.UE_label == 1:
            print('mew: ', mew)
            print('original_rho: ',original_rho)
            print('rho: ', rho)
        Pr_Lds = []
        queueing_violation_prob = 0
        
        # if self.Qd_lc > 0:
        for Ld_off in range(0,Ld_max+1):
            Pr_Ld = self.Pr_Ld_off(Ld_off)
            Pr_Lds.append(Pr_Ld)
        #print('len(Pr_Lds): ', Pr_Lds)
        sum_Pr_Lds = sum(Pr_Lds)

        # print('self.Pr_Qs:')
        # print(self.Pr_Qs)

        queueing_violation_prob = 1 - sum_Pr_Lds


        self.offload_queue_delay_violation_probability_ = queueing_violation_prob
        if self.UE_label == 1:
            print('offload queueing_violation_prob: ',queueing_violation_prob,'sum(sum_Pr_Lds): ', sum_Pr_Lds)
            print('---------------------------------------------------------------------------------')
        if queueing_violation_prob > 1:
            queueing_violation_prob = 1
        elif queueing_violation_prob < 0:
            queueing_violation_prob = 0

        #print('local queueing_violation_prob: ', queueing_violation_prob)
        #print('---------------------------------------------------------------------')
        queueing_violation_prob_reward = 0
        if (self.local_queue_delay_violation_probability_constraint-(queueing_violation_prob)) < 0:
            queueing_violation_prob_reward = (self.local_queue_delay_violation_probability_constraint-(queueing_violation_prob))
            self.offload_time_delay_violation_prob_constraint_violation_count = 1
        else:
            queueing_violation_prob_reward = 1
            self.offload_time_delay_violation_prob_constraint_violation_count = 0
        self.offload_queueing_violation_prob_reward = queueing_violation_prob_reward
        return queueing_violation_prob_reward
    


    







    
    




        

        






  



            