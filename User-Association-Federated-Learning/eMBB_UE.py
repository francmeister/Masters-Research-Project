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
        self.current_associated_access_point = 0
        self.user_association_channel_rate = 0
        self.distance_from_associated_access_point = 0
        self.x_coordinate = 0
        self.y_coordinate = 0
        self.user_association_channel_rate_array = []
        self.distances_from_access_points = []
        self.access_points_channel_rates = []
        self.set_properties_eMBB()

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.x_coordinate = coordinates[0]
        self.y_coordinate = coordinates[1]

    def generate_unique_numbers(self,limit):
        numbers  = list(range(0, limit))
        random_numbers = []
        for _ in range(3):
            # Shuffle the remaining numbers
            random.shuffle(numbers)
            # Select the first number
            number = numbers.pop()
            random_numbers.append(number)
        return random_numbers

    def calculate_distances_from_access_point(self,access_points_coordinates, radius):
        self.distances_from_access_point = []
        self.access_points_within_radius = []
        self.distances_from_access_points = []
        #print(access_points_coordinates)
        access_point_number = 1
        for access_point_coordinate in access_points_coordinates:
            distance_from_access_point = self.calculate_distance_from_access_point(access_point_coordinate)
            self.distances_from_access_point.append(distance_from_access_point)
            self.distances_from_access_points.append((self.user_label,access_point_number,distance_from_access_point))
            access_point_number+=1
        
        # num_access_points = len(self.distances_from_access_point)
        # random_nums = self.generate_unique_numbers(num_access_points)

        # first_rand_num = random_nums[0]
        # second_rand_num = random_nums[1]
        # third_rand_num = random_nums[2]

        # self.distances_from_access_point[first_rand_num] = self.distances_from_access_point[first_rand_num]/10000000000
        # self.distances_from_access_point[second_rand_num] = self.distances_from_access_point[second_rand_num]/100000
        # self.distances_from_access_point[third_rand_num] = self.distances_from_access_point[third_rand_num]*10000
        # #print('distances_from_access_point: ', self.distances_from_access_point)
        #print('')
        #print('user: ', self.user_label, 'self.distances_from_access_point: ', self.distances_from_access_point)
        access_point_number = 1
        for distance_from_access_point in self.distances_from_access_point:
            if distance_from_access_point <= radius:
                self.access_points_within_radius.append((access_point_number,distance_from_access_point))
            access_point_number+=1

        #print(self.distances_from_access_point)

    def calculate_distance_from_access_point(self,access_point_coordinate):
        #print(access_point_coordinate[0])
        #print(math.pow((self.y_coordinate-access_point_coordinate[1]),2))
        distance_squared = math.pow((self.x_coordinate-access_point_coordinate[0]),2) + math.pow((self.y_coordinate-access_point_coordinate[1]),2)
        #print('distance_squared: ', distance_squared)
        distance = math.sqrt(distance_squared)
        return distance

    def calculate_user_association_channel_gains(self):
        self.fast_fading_channel_gain =  np.random.exponential(1)

        if self.slow_fading_gain_change_timer == 0:
            g_l = np.random.normal(loc=0, scale=8, size=1)
            #print('eMBB User: ', self.user_label,'self.distance_from_associated_access_point in timer function: ', self.distance_from_associated_access_point)
            g = 35.3 + 37.8 * np.log10(self.distance_from_associated_access_point) + g_l
            #print('eMBB User: ', self.user_label,'g: ', g)
            G = 10 ** (-g/10)
            #print('eMBB User: ', self.user_label,'G: ', G)
            self.slow_fading_channel_gain = G#np.random.exponential(1) 
            self.slow_fading_gain_change_timer = 0

        #print('self.slow_fading_gain_change_timer: ', self.slow_fading_gain_change_timer)
        self.slow_fading_gain_change_timer+=1

        #return self.fast_fading_channel_gain*self.slow_fading_channel_gain
    
    def calculate_achieved_user_association_channel_rate(self, communication_channel):
        #self.user_association_channel_rate = math.pow(self.distance_from_associated_access_point,-1)#*self.fast_fading_channel_gain*self.slow_fading_channel_gain
        #print('eMBB User: ', self.user_label,'self.slow_fading_channel_gain: ', self.slow_fading_channel_gain)
        #print('self.fast_fading_channel_gain: ', self.fast_fading_channel_gain)
        RB_channel_gain = self.slow_fading_channel_gain*self.fast_fading_channel_gain
        #print('RB_channel_gain: ', RB_channel_gain)
        RB_bandwidth = communication_channel.system_bandwidth_Hz_user_association
        noise_spectral_density = communication_channel.noise_spectral_density_W
        channel_rate_numerator = self.max_transmission_power_W*RB_channel_gain
        channel_rate_denominator = noise_spectral_density*RB_bandwidth
        channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
        self.user_association_channel_rate = channel_rate
        self.user_association_channel_rate_array.append(self.user_association_channel_rate)
        self.user_association_channel_rate = sum(self.user_association_channel_rate_array)/len(self.user_association_channel_rate_array)
        #random_value = 0.0001*random.random()
        #print('channel_rate_numerator: ', channel_rate_numerator)
        #print('channel_rate_denominator: ', channel_rate_denominator)
        #print(self.user_association_channel_rate)
        #print('embb: ', self.user_label, 'user association channel rate: ', self.user_association_channel_rate)
        return self.user_association_channel_rate#random_value*math.pow(self.distance_from_associated_access_point,-1)*10000#self.user_association_channel_rate
    

    def calculate_channel_rate_to_other_access_points(self, communication_channel, steps,step_limit):
        #self.access_points_channel_rates = []
        #self.ap_slot_channel_rates = []
        #print('user: ', self.user_label,'self.distances_from_access_point: ', self.distances_from_access_point)
        for distance_from_access_point in self.distances_from_access_point:
            fast_fading_gain = np.random.exponential(1)
            g_l = np.random.normal(loc=0, scale=8, size=1)
            g = 35.3 + 37.8 * np.log10(distance_from_access_point) + g_l
            G = 10 ** (-g/10)
            slow_fading_gain = G
            RB_channel_gain = fast_fading_gain*slow_fading_gain
            RB_bandwidth = communication_channel.system_bandwidth_Hz_user_association
            noise_spectral_density = communication_channel.noise_spectral_density_W
            channel_rate_numerator = self.max_transmission_power_W*RB_channel_gain
            channel_rate_denominator = noise_spectral_density*RB_bandwidth
            channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
            self.ap_slot_channel_rates.append(channel_rate)
            print('eMBB user: ', self.eMBB_UE_label, 'steps: ', steps,'slot channel rates: ', self.ap_slot_channel_rates)

        if steps == step_limit:
            number_of_access_points = len(self.distances_from_access_point)
            number_of_slot = len(self.ap_slot_channel_rates)/number_of_access_points
            print('eMBB user: ', self.eMBB_UE_label, 'number_of_access_points: ', number_of_access_points, 'number_of_slot: ', number_of_slot, 'len(self.ap_slot_channel_rates): ',len(self.ap_slot_channel_rates))
            self.ap_slot_channel_rates = np.array(self.ap_slot_channel_rates)
            print('eMBB user: ', self.eMBB_UE_label, 'np array self.ap_slot_channel_rates: ', self.ap_slot_channel_rates, 'self.ap_slot_channel_rates shape: ', self.ap_slot_channel_rates.shape)
            self.ap_slot_channel_rates = self.ap_slot_channel_rates.reshape(int(number_of_slot), number_of_access_points)
            print('eMBB user: ', self.eMBB_UE_label, 'slot channel rates reshaped: ', self.ap_slot_channel_rates)
            average_channel_rates = np.mean(self.ap_slot_channel_rates, axis=0)
            print('eMBB user: ', self.eMBB_UE_label, 'slot channel rates averages: ', average_channel_rates)
            access_point_number = 1
            for distance_from_access_point in self.distances_from_access_point:
                self.access_points_channel_rates.append((self.user_label, access_point_number, average_channel_rates[access_point_number-1]))
                access_point_number+=1
       # return self.user_association_channel_rate*100
        #print('user: ', self.user_label,'self.access_points_channel_rates: ', self.access_points_channel_rates)

    def calculate_distance_from_current_access_point(self):
        #print('embb user: ', self.user_label, 'current_associated_access_point: ', self.current_associated_access_point, 'distances_from_access_point: ', self.distances_from_access_point)
        #print(self.distances_from_access_point)
        # if timestep < 8000:
        #     self.distance_from_associated_access_point = max(self.distances_from_access_point)

        # elif timestep >= 8000 and timestep < 10000:
        #     self.distances_from_access_point.sort(reverse=True)
        #     rand_num = random.randint(0,1)
        #     self.distance_from_associated_access_point = self.distances_from_access_point[rand_num]

        # elif timestep >= 10000 and timestep < 12000:
        #     self.distances_from_access_point.sort(reverse=True)
        #     rand_num = random.randint(1,2)
        #     self.distance_from_associated_access_point = self.distances_from_access_point[rand_num]

        # elif timestep >= 12000:
        #     self.distance_from_associated_access_point = min(self.distances_from_access_point)

        #print('eMBB User: ', self.user_label, 'self.current_associated_access_point: ', self.current_associated_access_point)
        self.distance_from_associated_access_point = self.distances_from_access_point[self.current_associated_access_point-1]
        #print('eMBB User: ', self.user_label,'self.distance_from_associated_access_point: ', self.distance_from_associated_access_point)
        #print('eMBB User: ', self.user_label,'self.distances_from_access_point: ', self.distances_from_access_point)
        #self.distance_from_associated_access_point = max(self.distances_from_access_point)#self.distances_from_access_point[self.current_associated_access_point-1]
        #print('embb user: ', self.eMBB_UE_label, 'distance from associated AP: ', self.distance_from_associated_access_point)


    def set_properties_eMBB(self):
        #self.access_points_channel_rates = []
        self.ap_slot_channel_rates = []
        self.distances_from_access_point = []
        self.slow_fading_gain_change_timer = 0
        self.fast_fading_channel_gain =  np.random.exponential(1)
        self.slow_fading_channel_gain = np.random.exponential(1)
        self.user_association_channel_gain = 0
        #State Space Limits
        self.max_allowable_latency = 2000 #[1,2] s
        self.min_allowable_latency = 1000

        self.max_allowable_reliability = 0

        self.min_communication_qeueu_size = 0
        self.max_communication_qeueu_size = 50

        self.min_channel_gain = math.pow(10,-5)
        self.max_channel_gain = 10

        self.min_energy_harvested = 0
        self.max_energy_harvested = 150

        self.max_battery_energy = 2000#22000
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

        self.max_lc_queue_length = 50
        self.max_off_queue_length = 400

        self.min_lc_queue_length = 0
        self.min_off_queue_length = 0

        self.battery_energy_level = 20000#(random.randint(15000,25000))
        self.energy_harvesting_constant = 300
        self.cycles_per_byte = 330
        self.cycles_per_bit = self.cycles_per_byte/8
        self.max_service_rate_cycles_per_slot = 620000
        self.service_rate_bits_per_slot = (self.max_service_rate_cycles_per_slot/self.cycles_per_byte)*8
        

        #self.QOS_requirement = QOS_requirement()
        #self.QOS_requirement_for_transmission = QOS_requirement()
        #self.user_task = Task(330)
        #self.local_task = Task(330)
        #self.offload_task = Task(330)
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
        self.energy_consumption_coefficient = math.pow(10,-13.8)
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

        self.max_transmission_power_dBm = 400 # dBm
        self.min_transmission_power_dBm = 0
        self.max_transmission_power_W =  100*10**(-3)#(math.pow(10,(self.max_transmission_power_dBm/10)))/1000# Watts
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
        self.task_arrival_rate_tasks_per_second = 0
        self.ptr = 0
        self.queuing_delay = 0
        self.previous_slot_battery_energy = 0

        self.total_gain = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound*2)
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
        self.task_arrival_rate = 0
        self.offloading_ratio = 0
        self.average_packet_size_bits = 0
        self.max_lc_queue_delay_violation_probability = 0.8
        self.user_association_channel_rate_array = []


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
        self.task_arrival_rate_tasks_per_second = np.random.poisson(25,1)#np.random.poisson(5,1)
        self.task_arrival_rate_tasks_per_second = self.task_arrival_rate_tasks_per_second[0]
        self.previous_arrival_rate = self.task_arrival_rate_tasks_per_second
        self.current_arrival_rate = self.task_arrival_rate_tasks_per_second
        qeueu_timer = 0


        if len(self.task_queue) >= self.max_queue_length_number:
            for x in range(0,self.task_arrival_rate_tasks_per_second):
                #np.random.poisson(10)
                #task_size_per_second_kilobytes = random.randint(self.min_task_size_KB_per_second,self.max_task_size_KB_per_second) #choose between 50 and 100 kilobytes
                #task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*self.task_arrival_rate_tasks_per_second
                #task_size_per_slot_kilobytes = task_size_per_second_kilobytes*task_arrival_rate_tasks_slot
                task_size_per_slot_bits = int(np.random.uniform(500,1500))#Average of 1000 bits per task in slot #int(task_size_per_slot_kilobytes*8000) #8000 bits in a KB----------
                self.previous_task_size_bits = task_size_per_slot_bits
                #task_cycles_required = self.cycles_per_bit*task_size_per_slot_bits#-------------
                latency_requirement = 10#latency required is 10 ms for every task#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
                reliability_requirement = 0
                QOS_requirement_ = QOS_requirement(latency_requirement,reliability_requirement)
                user_task = Task(330,task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
                self.task_identifier+=1
                #print('task identifier: ', self.task_identifier)

                self.storage[int(self.ptr)] = user_task
                self.ptr = (self.ptr + 1) % self.max_queue_length_number
        else:
            for x in range(0,self.task_arrival_rate_tasks_per_second):
                #task_size_per_second_kilobytes = random.randint(self.min_task_size_KB_per_second,self.max_task_size_KB_per_second) #choose between 50 and 100 kilobytes
                #task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*self.task_arrival_rate_tasks_per_second
                #task_size_per_slot_kilobytes = task_size_per_second_kilobytes*task_arrival_rate_tasks_slot
                task_size_per_slot_bits = int(np.random.uniform(500,1500)) #8000 bits in a KB----------
                self.previous_task_size_bits = task_size_per_slot_bits
                #task_cycles_required = self.cycles_per_bit*task_size_per_slot_bits#-------------
                latency_requirement = 10#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
                reliability_requirement = 0
                QOS_requirement_ = QOS_requirement(latency_requirement,reliability_requirement)
                user_task = Task(330,task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
                self.task_identifier+=1
                #print('task identifier: ', self.task_identifier)
                self.task_queue.append(user_task)
        

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos, Env_width_pixels, Env_width_metres):

        x_diff_metres = abs(SBS_x_pos-self.x_position)
        y_diff_metres = abs(SBS_y_pos-self.y_position)


        self.distance_from_SBS = math.sqrt(math.pow(x_diff_metres,2)+math.pow(y_diff_metres,2))

    def collect_state(self):
        #self.cpu_clock_frequency = (random.randint(5,5000))
        offloading_queue_length, local_queue_length = self.calculate_queue_lengths()
        self.user_state_space.collect(self.total_gain.squeeze(),self.previous_slot_battery_energy,offloading_queue_length, local_queue_length)
        #self.user_state_space.collect(self.total_gain,self.communication_queue,self.battery_energy_level,self.communication_queue[0].QOS_requirement,self.cpu_clock_frequency)
        return self.user_state_space

    def split_tasks(self):
        if len(self.task_queue) > 0:
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

            data = {
                'Task Identity':task_identities,
                'Task Size Bits':task_sizes_bits,
                #'Required Cycles':required_cycles,
                'Latency requirement':latency_requirements
            }

            df = pd.DataFrame(data=data)
            # print('--------------------------------------------Timeslot: ',self.timeslot_counter, '--------------------------------------------')
            # print('task queue data')
            # print(df)
            # print(' ')

            # print('self.allocated_offloading_ratio')
            # print(self.allocated_offloading_ratio)
            # print('')
            for x in range(0,self.task_arrival_rate_tasks_per_second):
                #print('self.task_queue[x]: ', len(self.task_queue))
                packet_dec = self.task_queue[x].bits
                self.QOS_requirement_for_transmission = self.task_queue[x].QOS_requirement
                packet_bin = bin(packet_dec)[2:]
                packet_size = len(packet_bin)
                self.packet_offload_size_bits = int(self.allocated_offloading_ratio*packet_size)
                self.packet_local_size_bits = int((1-self.allocated_offloading_ratio)*packet_size)

                if self.packet_local_size_bits > 0:
                    local_task = Task(330,self.packet_local_size_bits,self.task_queue[x].QOS_requirement,self.task_queue[x].queue_timer,self.task_queue[x].task_identifier)
                    self.local_queue.append(local_task)

                if self.packet_offload_size_bits > 0:
                    offload_task = Task(330,self.packet_offload_size_bits,self.task_queue[x].QOS_requirement,self.task_queue[x].queue_timer,self.task_queue[x].task_identifier)
                    self.communication_queue.append(offload_task)

                
                

                #self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
                #self.has_transmitted_this_time_slot = True
            for x in range(0,self.task_arrival_rate_tasks_per_second):
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

            if len(self.local_queue) > 0:
                for task in self.local_queue:
                    local_task_identities.append(task.task_identifier)
                    local_task_sizes_bits.append(task.slot_task_size)
                    local_required_cycles.append(task.required_computation_cycles)
                    local_latency_requirements.append(task.QOS_requirement.max_allowable_latency)

            local_data = {
                'Task Identity':local_task_identities,
                'Task Size Bits':local_task_sizes_bits,
                #'Required Cycles':local_required_cycles,
                'Latency requirement':local_latency_requirements
            }

            df = pd.DataFrame(data=local_data)
            # print('local queue data')
            # print(df)
            # print(' ')

            offload_task_identities = []
            offload_task_sizes_bits = []
            offload_required_cycles = []
            offload_latency_requirements = []

            if len(self.communication_queue) > 0:
                for task in self.communication_queue:
                    offload_task_identities.append(task.task_identifier)
                    offload_task_sizes_bits.append(task.slot_task_size)
                    offload_required_cycles.append(task.required_computation_cycles)
                    offload_latency_requirements.append(task.QOS_requirement.max_allowable_latency)

            offload_data = {
                'Task Identity':offload_task_identities,
                'Task Size Bits':offload_task_sizes_bits,
                #'Required Cycles':offload_required_cycles,
                'Latency requirement':offload_latency_requirements
            }

            df = pd.DataFrame(data=offload_data)
            # print('offload queue data')
            # print(df)
            # print(' ')
            # print('Size of offloading queue: ',sum(offload_task_sizes_bits))
       

    def transmit_to_SBS(self, communication_channel, URLLC_users):
        #Calculate the bandwidth achieved on each RB
        achieved_RB_channel_rates = []
        # achieved_RB_channel_rates_ = []
        #print('number of allocated RBs: ', len(self.allocate(d_RBs))
        count = 0
        self.find_puncturing_users(communication_channel,URLLC_users)
        #print('allocated RBs')
        #print(self.allocated_RBs)
        reshaped_allocated_RBs = np.array(self.allocated_RBs)
        reshaped_allocated_RBs = reshaped_allocated_RBs.squeeze()#.reshape(1,communication_channel.time_divisions_per_slot*communication_channel.num_allocate_RBs_upper_bound)
        reshaped_allocated_RBs = reshaped_allocated_RBs.reshape(communication_channel.time_divisions_per_slot,communication_channel.num_allocate_RBs_upper_bound)
        
        if self.battery_energy_level > 0 and self.has_transmitted_this_time_slot == True:
            for tb in range(0,communication_channel.time_divisions_per_slot):
                for rb in range(0,communication_channel.num_allocate_RBs_upper_bound):
                    RB_indicator = reshaped_allocated_RBs[tb][rb]
                    current_rb_occupied = False
                    for occupied_resource_time_block in self.occupied_resource_time_blocks:
                        #print('occupied_resource_time_block: ', occupied_resource_time_block)
                        if occupied_resource_time_block[0] == tb+1 and occupied_resource_time_block[1] == rb+1 and occupied_resource_time_block[2] == 1:
                            current_rb_occupied = True
                            break
                        elif occupied_resource_time_block[0] == tb+1 and occupied_resource_time_block[1] == rb+1 and occupied_resource_time_block[2] == 0:
                            current_rb_occupied = False
                            break
                    # print('tb: ', tb+1, ' rb: ', rb+1, ' currently occupied: ', current_rb_occupied)
                    # print('')
                    #if RB_indicator == 1:
                    RB_channel_gain = self.total_gain[0][rb]
                    achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied)
                    #achieved_RB_channel_rate_ = self.calculate_channel_rate_(communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied)
                    achieved_RB_channel_rates.append(achieved_RB_channel_rate)
                    #achieved_RB_channel_rates_.append(achieved_RB_channel_rate_)

            self.achieved_channel_rate = sum(achieved_RB_channel_rates)
            #self.achieved_channel_rate_ = sum(achieved_RB_channel_rates_)
            self.previous_channel_rate = self.achieved_channel_rate
            min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
            self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,7000],[0,1]) 
        # print('achieved channel rate: ', self.achieved_channel_rate)
        # print('achieved channel rate_: ', self.achieved_channel_rate_)
        # print('')  
                        







        # if self.battery_energy_level > 0 and self.has_transmitted_this_time_slot == True:
        #      for RB_indicator in self.allocated_RBs:
        #          RB_channel_gain = self.total_gain[0][count]
        #          achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel,RB_indicator,RB_channel_gain)
        #          achieved_RB_channel_rates.append(achieved_RB_channel_rate)
        #          count += 1
        #      self.achieved_channel_rate = sum(achieved_RB_channel_rates)
        #      self.previous_channel_rate = self.achieved_channel_rate
        #      min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        #      self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,7000],[0,1])   
        # # 
         

        

        #print('achieved channel rate: ', self.achieved_channel_rate)
        #print(' ')

    def calculate_channel_rate(self, communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied):
        RB_bandwidth = communication_channel.RB_bandwidth_Hz
        noise_spectral_density = communication_channel.noise_spectral_density_W
        channel_rate_numerator = self.assigned_transmit_power_W*RB_channel_gain
        channel_rate_denominator = noise_spectral_density#*RB_bandwidth
        half_num_mini_slots_per_rb = communication_channel.num_of_mini_slots/2
        if current_rb_occupied == False:
            channel_rate = RB_indicator*(RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator)))
        elif current_rb_occupied == True:
            channel_rate = RB_indicator*RB_bandwidth*(1-(1/half_num_mini_slots_per_rb))*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
        return (channel_rate/1000)
    
    # def calculate_channel_rate_(self, communication_channel,RB_indicator,RB_channel_gain,current_rb_occupied):
    #     RB_bandwidth = communication_channel.RB_bandwidth_Hz
    #     noise_spectral_density = communication_channel.noise_spectral_density_W
    #     channel_rate_numerator = self.assigned_transmit_power_W*RB_channel_gain
    #     channel_rate_denominator = noise_spectral_density#*RB_bandwidth
    #     half_num_mini_slots_per_rb = communication_channel.num_of_mini_slots/2
    
    #     channel_rate = RB_indicator*(RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator)))
 
    #     return (channel_rate/1000)
    
    def local_processing(self):
        cpu_cycles_left = self.max_service_rate_cycles_per_slot #check if 
        self.achieved_local_energy_consumption = 0
        self.dequeued_local_tasks.clear()
        used_cpu_cycles = 0
        counter = 0

        for local_task in self.local_queue:
            #print('cycles left: ', cpu_cycles_left)
            #print('local_task.required_computation_cycles: ', local_task.required_computation_cycles)
            if cpu_cycles_left > local_task.required_computation_cycles:
                #print('cycles left: ', cpu_cycles_left)
                #self.achieved_local_energy_consumption += self.energy_consumption_coefficient*math.pow(local_task.required_computation_cycles,2)*local_task.required_computation_cycles
                cpu_cycles_left-=local_task.required_computation_cycles
                self.dequeued_local_tasks.append(local_task)
                counter += 1

            elif cpu_cycles_left < local_task.required_computation_cycles and cpu_cycles_left > self.cycles_per_bit:
                bits_that_can_be_processed = cpu_cycles_left/self.cycles_per_bit
                #self.achieved_local_energy_consumption += self.energy_consumption_coefficient*math.pow(cpu_cycles_left,2)*cpu_cycles_left
                local_task.split_task(bits_that_can_be_processed) 
                break

        for x in range(0,counter):
            self.local_queue.pop(0)
        #self.energy_consumption_coefficient*math.pow(self.max_service_rate_cycles_per_slot,2) = energy consumed per cycle (J/cycle)
        used_cpu_cycles = self.max_service_rate_cycles_per_slot - cpu_cycles_left
        self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.max_service_rate_cycles_per_slot,2)*used_cpu_cycles
        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        dequeued_task_size = []
        total_sum_size_dequeued_tasks = []

        if len(self.dequeued_local_tasks) > 0:
            for dequeued_local_task in self.dequeued_local_tasks:
                task_identities.append(dequeued_local_task.task_identifier)
                task_latency_requirements.append(dequeued_local_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(dequeued_local_task.queue_timer)
                dequeued_task_size.append(dequeued_local_task.slot_task_size)

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

            # print('Dequeued Local Tasks')
            # print(df)
            # print(' ')
            # print('Local Computation energy consumed in this slot: ', self.achieved_local_energy_consumption)
            # print(' ')
            # print('-----------------dequeued local tasks size total---------------------')
            #print(total_sum_size_dequeued_tasks)
            #print('')

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
        #print('achieved channel rate')
        #print(self.achieved_channel_rate)
        if self.achieved_channel_rate == 0:
            self.achieved_transmission_delay = 0
        else:
            left_bits = communication_channel.long_TTI*self.achieved_channel_rate
            for offloading_task in self.communication_queue:
                if offloading_task.slot_task_size < left_bits:
                    offloading_bits += offloading_task.slot_task_size
                    left_bits -= offloading_task.slot_task_size
                    self.dequeued_offload_tasks.append(offloading_task)
                    counter+=1

                elif offloading_task.slot_task_size > left_bits:
                    offloading_task.split_task(left_bits)
                    offloading_bits+=left_bits

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

        if len(self.dequeued_offload_tasks) > 0:
            for dequeued_offload_task in self.dequeued_offload_tasks:
                task_identities.append(dequeued_offload_task.task_identifier)
                task_latency_requirements.append(dequeued_offload_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(dequeued_offload_task.queue_timer)
                achieved_throughput.append(self.achieved_channel_rate)
                number_of_allocated_RBs.append(len(self.allocated_RBs))
                task_sizes.append(dequeued_offload_task.slot_task_size)
                

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

            # print('Dequeued Offload Tasks')
            # print(df)
            # print(' ')
            # print('Achieved TTI channel rate: ', self.achieved_channel_rate)
            # print(' ')

        self.check_completed_tasks()
        #self.achieved_transmission_delay = 1
        self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*(1/communication_channel.time_divisions_per_slot)*sum(self.allocated_RBs)
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
            if local_dequeued_task.task_identifier not in offload_queue_task_identities and local_dequeued_task.task_identifier not in dequeued_offload_task_identities:
                self.completed_tasks.append(local_dequeued_task)

        for offload_dequeued_task in self.dequeued_offload_tasks:
            if offload_dequeued_task.task_identifier not in local_queue_task_identities:
                self.completed_tasks.append(offload_dequeued_task) 

        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        if len(self.completed_tasks) > 0:
            for completed_task in self.completed_tasks:
                task_identities.append(completed_task.task_identifier)
                task_latency_requirements.append(completed_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(completed_task.queue_timer)              

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency
            }


            df = pd.DataFrame(data=data)

            #print('Completed Tasks')
            #print(df)
            #print(' ')
            #print(' ')

        sum_latency = 0
        for completed_task in self.completed_tasks:
            if completed_task.QOS_requirement.max_allowable_latency < completed_task.queue_timer:
                sum_latency+= (completed_task.QOS_requirement.max_allowable_latency - completed_task.queue_timer)

        self.queuing_delay = sum_latency
        #print('self.queuing_delay', self.queuing_delay)
        

    def total_energy_consumed(self):
        #print(self.battery_energy_level)
        if self.battery_energy_level >  self.achieved_total_energy_consumption:
            self.achieved_total_energy_consumption = self.achieved_local_energy_consumption + self.achieved_transmission_energy_consumption
            self.achieved_total_energy_consumption_normalized = interp(self.achieved_total_energy_consumption,[0,5500],[0,1])
            self.battery_energy_level = self.battery_energy_level - self.achieved_total_energy_consumption
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
        self.small_scale_channel_gain = small_scale_gain
        first_large_scale_gain = large_scale_gain[0][0]
        item = 0
        for gain in large_scale_gain[0]:
            large_scale_gain[0][item] = first_large_scale_gain
            item+=1

        self.small_scale_gain = small_scale_gain
        self.large_scale_gain = large_scale_gain
    
        #print('small_scale_gain')
        #print(small_scale_gain)
        #print('larger_scale_gain')
        #print(large_scale_gain)
        self.total_gain = np.concatenate((small_scale_gain,large_scale_gain),axis=1)#np.random.exponential(1,size=(1,number_of_RBs))
        #print('self.total_gain')
        #print(self.total_gain)
        #self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        #self.total_gain = self.small_scale_channel_gain#*self.large_scale_channel_gain#self.pathloss_gain
        #if self.total_gain < 0.1:
        #    self.total_gain = 0.1

    def calculate_assigned_transmit_power_W(self):
        self.assigned_transmit_power_W = self.assigned_transmit_power_dBm#(math.pow(10,(self.assigned_transmit_power_dBm/10)))/1000

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

    def harvest_energy(self):
        self.energy_harvested = np.random.exponential(250)#random.randint(0,2000)
        small_scale_gain = self.small_scale_gain[0]
        large_scale_gain = self.large_scale_gain[0]
        total_gain = sum(small_scale_gain*large_scale_gain)
        self.energy_harvested = self.energy_harvesting_constant*total_gain


    def energy_consumption_reward(self):
        energy_reward = self.battery_energy_level #+ self.energy_harversted 

        max_energy_reward = 40000
        min_energy_reward = -10000

        #energy_reward_normalized = 0
        #if energy_reward >= 0:
        energy_reward_normalized = interp(energy_reward,[min_energy_reward,max_energy_reward],[0,1])
        #else:
        #    energy_reward_normalized = -0.2

        return energy_reward_normalized
    
    def increment_task_queue_timers(self):
        if len(self.task_queue) > 0:
            for task in self.task_queue:
                task.increment_queue_timer()

        if len(self.local_queue) > 0:
            for local_task in self.local_queue:
                local_task.increment_queue_timer()

        if len(self.communication_queue) > 0:
            for offload_task in self.communication_queue:
                offload_task.increment_queue_timer()

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
            offload_queuing_delay = -(self.current_queue_length_off/current_arrival_rate_off)
        else:
            offload_queuing_delay = 1  

        if self.current_queue_length_lc < 0:
            current_arrival_rate_lc = self.current_arrival_rate*(1-self.allocated_offloading_ratio)
            local_queuing_delay = -(self.current_queue_length_lc/current_arrival_rate_lc)
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
        # print('reshaped_allocated_RBs')
        # print(reshaped_allocated_RBs)
        # print('')
        reshaped_allocated_RBs = reshaped_allocated_RBs.squeeze()#.reshape(1,communication_channel.time_divisions_per_slot*communication_channel.num_allocate_RBs_upper_bound)
        reshaped_allocated_RBs = reshaped_allocated_RBs.reshape(communication_channel.time_divisions_per_slot,communication_channel.num_allocate_RBs_upper_bound)
        # print('reshaped_allocated_RBs')
        # print(reshaped_allocated_RBs)
        # print('')
        # print('')
        #print(reshaped_allocated_RBs)
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
       
        self.puncturing_urllc_users(URLLC_users)

    def puncturing_urllc_users(self,urllc_users):
        self.puncturing_urllc_users_.clear()
        self.occupied_resource_time_blocks.clear()
        for allocated_resource_block in self.allocated_resource_blocks_numbered:
            
            time_blocks_at_this_rb = self.time_matrix[allocated_resource_block-1]
            if time_blocks_at_this_rb == 1 or time_blocks_at_this_rb == 2:
                for urllc_user in urllc_users:
                    if urllc_user.assigned_resource_block == allocated_resource_block and urllc_user.assigned_time_block == time_blocks_at_this_rb:
                        self.puncturing_urllc_users_.append(urllc_user.URLLC_UE_label)
                        if urllc_user.has_transmitted_this_time_slot == True:
                            self.occupied_resource_time_blocks.append((time_blocks_at_this_rb,allocated_resource_block,1))
                        elif urllc_user.has_transmitted_this_time_slot == False:
                            self.occupied_resource_time_blocks.append((time_blocks_at_this_rb,allocated_resource_block,0))

            
            elif time_blocks_at_this_rb == (1,2):
                for time_block_at_this_rb in time_blocks_at_this_rb:
                    for urllc_user in urllc_users:
                        if urllc_user.assigned_resource_block == allocated_resource_block and urllc_user.assigned_time_block == time_block_at_this_rb:
                            self.puncturing_urllc_users_.append(urllc_user.URLLC_UE_label)
                            if urllc_user.has_transmitted_this_time_slot == True:
                                self.occupied_resource_time_blocks.append((time_block_at_this_rb,allocated_resource_block,1))
                            elif urllc_user.has_transmitted_this_time_slot == False:
                                self.occupied_resource_time_blocks.append((time_block_at_this_rb,allocated_resource_block,0))

        # print('émbb user id: ', self.eMBB_UE_label, 'allocated rb: ', self.allocated_resource_blocks_numbered)
        # print('allocated time blocks: ', self.time_matrix)
        # print('')
        # print('self.puncturing_urllc_users_: ', self.puncturing_urllc_users_)
        # print('occupied resource blocks: ', self.occupied_resource_time_blocks)
        # print('')

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
        if len(self.local_queue) > 0:
            for task in self.local_queue:
                average_task_cycles_per_packet+=task.required_computation_cycles
            average_task_cycles_per_packet = average_task_cycles_per_packet/len(self.local_queue)

        local_computation_time = average_task_cycles_per_packet/self.max_service_rate_cycles_per_slot
        local_queueing_time = len(self.local_queue)*local_computation_time
        local_delay = local_computation_time+local_queueing_time

        average_packet_size_bits = 0
        if len(self.communication_queue) > 0:
            for task in self.communication_queue:
                average_packet_size_bits+=task.slot_task_size
            average_packet_size_bits =  average_packet_size_bits/len(self.communication_queue)

        expected_rate_over_prev_T_slot = self.embb_rate_expectation_over_prev_T_slot(5,self.achieved_channel_rate)
        if expected_rate_over_prev_T_slot > 0:
            offload_queueing_time = average_packet_size_bits/expected_rate_over_prev_T_slot
        else:
            offload_queueing_time = 0
        offloading_delay = offload_queueing_time + 1

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
        offload_traffic = 0
        if self.achieved_channel_rate > 0:
            offload_traffic = (self.allocated_offloading_ratio*self.task_arrival_rate*self.average_packet_size_bits)/self.achieved_channel_rate
        else:
            offload_traffic
        reward = 1-offload_traffic
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

        
    




        

        






  



            