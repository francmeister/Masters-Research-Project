import pygame, sys, time, random
import random
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
import numpy as np
from matplotlib.patches import Rectangle
import math
from State_Space import State_Space
from numpy import interp

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,x,y):
        #User_Equipment.__init__(self)
        self.UE_label = eMBB_UE_label
        self.original_x_position = x
        self.original_y_position = y
        self.eMBB_UE_label = eMBB_UE_label
        self.set_properties_eMBB()

    def set_properties_eMBB(self):

        #State Space Limits
        self.max_allowable_latency = 2000 #[1,2] s
        self.min_allowable_latency = 0

        self.max_allowable_reliability = 0

        self.min_communication_qeueu_size = 0
        self.max_communication_qeueu_size = 5000*8000

        self.min_channel_gain = 0.1
        self.max_channel_gain = 10

        self.min_energy_harvested = 0
        self.max_energy_harvested = 100

        self.max_battery_energy = 25000
        self.min_battery_energy = 0

        self.max_cpu_frequency = 5000
        self.min_cpu_frequency = 5

        self.battery_energy_level = (random.randint(15000,25000))

        self.QOS_requirement = QOS_requirement()
        self.QOS_requirement_for_transmission = QOS_requirement()
        self.user_task = Task(330)
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
        self.total_gain = 0
        self.has_transmitted_this_time_slot = False
        self.communication_queue = []
        self.energy_consumption_coefficient = math.pow(10,-15)
        self.achieved_transmission_energy_consumption = 0
        self.achieved_local_processing_delay = 0
        self.achieved_total_energy_consumption = 0
        self.achieved_total_processing_delay = 0
        self.cpu_cycles_per_byte = 330
        self.cpu_clock_frequency = (random.randint(5,5000)) #cycles/slot
        self.user_state_space = State_Space(self.UE_label,self.total_gain,self.communication_queue,self.battery_energy_level,self.QOS_requirement,self.cpu_clock_frequency)
        self.allocated_offloading_ratio = 0
        self.packet_offload_size_bits = 0
        self.packet_local_size_bits = 0
        self.packet_size = 0
        self.delay_reward = 10
        self.achieved_channel_rate_normalized = 0
        self.achieved_total_energy_consumption_normalized = 0
   
    
        self.single_side_standard_deviation_pos = 5
        self.xpos_move_lower_bound = self.x_position - self.single_side_standard_deviation_pos
        self.xpos_move_upper_bound = self.x_position + self.single_side_standard_deviation_pos
        self.ypos_move_lower_bound = self.y_position - self.single_side_standard_deviation_pos
        self.ypos_move_upper_bound = self.y_position + self.single_side_standard_deviation_pos
        self.allocated_RBs = []

        self.max_transmission_power_dBm = 70 # dBm
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

    def move_user(self,ENV_WIDTH,ENV_HEIGHT):
        self.x_position = random.randint(self.xpos_move_lower_bound,self.xpos_move_upper_bound)
        self.y_position = random.randint(self.ypos_move_lower_bound,self.ypos_move_upper_bound)

        if self.x_position < 0 or self.x_position > ENV_WIDTH:
            self.x_position = self.original_x_position

        if self.y_position < 0 or self.y_position > ENV_HEIGHT:
            self.y_position = self.original_y_position
        

    def generate_task(self,long_TTI):
        self.has_transmitted_this_time_slot = False
        self.timeslot_counter+=1
        
        #After every 1s generate packets with a uniform distribution between 5 and 10 packets per second
        #Long TTI = 1 ms. 1 second should be achieved after every 8000 timeslots
        if(self.timeslot_counter*long_TTI >= 1):
            self.timeslot_counter = 0
            #Require Task Arrival Rate, bits/packets, CPU cycles/packet
            self.task_arrival_rate_packets_per_second = random.randint(5,10) #Packets/s
            self.allowable_latency = random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
            self.allowable_reliability = 0
            self.packet_size = (random.randint(1000,5000))*8000 # [50,100]Kilobytes. 8000 bits in a KB
            self.QOS_requirement.set_requirements(self.allowable_latency,self.allowable_reliability)
            self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)
            self.communication_queue.append(self.user_task)
        

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos, Env_width_pixels, Env_width_metres):

        x_diff_metres = abs(SBS_x_pos-self.x_position)
        y_diff_metres = abs(SBS_y_pos-self.y_position)


        self.distance_from_SBS = math.sqrt(math.pow(x_diff_metres,2)+math.pow(y_diff_metres,2))

    def collect_state(self):
            self.cpu_clock_frequency = (random.randint(5,5000))
            self.user_state_space.collect(self.total_gain,self.communication_queue,self.battery_energy_level,self.QOS_requirement,self.cpu_clock_frequency)
            return self.user_state_space

    def split_packet(self):
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                packet_dec = self.communication_queue[0].packet_queue[0]
                self.QOS_requirement_for_transmission = self.communication_queue[0].QOS_requirement
                packet_bin = bin(packet_dec)[2:]
                packet_size = len(packet_bin)
                self.packet_offload_size_bits = int(self.allocated_offloading_ratio*packet_size)
                self.packet_local_size_bits = int((1-self.allocated_offloading_ratio)*packet_size)
                self.local_queue.append(random.getrandbits(self.packet_local_size_bits))
                self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
                self.has_transmitted_this_time_slot = True
                self.dequeue_packet()
       

    def transmit_to_SBS(self, communication_channel):
        #Calculate the bandwidth achieved on each RB
        achieved_RB_channel_rates = []
        #print('number of allocated RBs: ', len(self.allocate(d_RBs))
        if (len(self.allocated_RBs)  > 0) and (self.battery_energy_level > 0):
            for RB in self.allocated_RBs:
                achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel)
                achieved_RB_channel_rates.append(achieved_RB_channel_rate)
            #print('channel gain: ', self.total_gain, " achieved channel rate matrix: ", achieved_RB_channel_rates)
            self.achieved_channel_rate = sum(achieved_RB_channel_rates)
            min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
            self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[min_achievable_rate,max_achievable_rate],[0,1])
        else:
            self.achieved_channel_rate = 0
            self.achieved_channel_rate_normalized = 0

        #print('achieved channel rate: ', self.achieved_channel_rate)
        #print(' ')

    def calculate_channel_rate(self, communication_channel):
        RB_bandwidth = communication_channel.RB_bandwidth_Hz
        noise_spectral_density = communication_channel.noise_spectral_density_W
        channel_rate_numerator = self.assigned_transmit_power_W*self.total_gain
        channel_rate_denominator = noise_spectral_density*RB_bandwidth
        channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
        return channel_rate
    
    def local_processing(self):
        cycles_per_bit = self.cpu_cycles_per_byte*8*(self.packet_local_size_bits)
        self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.cpu_clock_frequency,2)*cycles_per_bit
        #print('self.achieved_local_energy_consumption: ', self.achieved_local_energy_consumption)
        self.achieved_local_processing_delay = cycles_per_bit/self.cpu_clock_frequency
        self.local_queue.pop(0) 

        min_local_energy_consumption, max_local_energy_consumption = self.min_and_max_achievable_local_energy_consumption()
        min_local_computation_delay, max_local_computation_delay = self.min_max_achievable_local_processing_delay()
        #print('min local delay: ', min_local_computation_delay, ' max local delay: ', max_local_computation_delay)
        #self.achieved_local_energy_consumption = interp(self.achieved_local_energy_consumption,[min_local_energy_consumption,max_local_energy_consumption],[0,5000])
        self.achieved_local_processing_delay = interp(self.achieved_local_processing_delay,[min_local_computation_delay,max_local_computation_delay],[0,500])
        #print('')

    def offloading(self,communication_channel):
        if self.achieved_channel_rate == 0:
            self.achieved_transmission_delay = 0
        else:
            self.achieved_transmission_delay = self.packet_offload_size_bits/self.achieved_channel_rate
            
        self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*self.achieved_transmission_delay
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[0,12*math.pow(10,-5)],[0,100])
        #print('transmission energy consumed: ', self.achieved_transmission_energy_consumption)
        min_offload_energy_consumption, max_offload_energy_consumption = self.min_and_max_achievable_offload_energy_consumption(communication_channel)
        min_offloading_delay, max_offloading_delay = self.min_max_achievable_offload_delay(communication_channel)
        #print('min offload delay: ', min_offloading_delay, ' max offload delay: ', max_offloading_delay)
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[min_offload_energy_consumption,max_offload_energy_consumption],[0,5000])
        self.achieved_transmission_delay = interp(self.achieved_transmission_delay,[min_offloading_delay,max_offloading_delay],[0,5000])
        #print('offload delay: ', self.achieved_transmission_delay)
        #print('transmission energy consumed: ', self.achieved_transmission_energy_consumption)
        

    def total_energy_consumed(self):
        if self.battery_energy_level >  self.achieved_total_energy_consumption:
            self.achieved_total_energy_consumption = self.achieved_local_energy_consumption + self.achieved_transmission_energy_consumption
            self.achieved_total_energy_consumption_normalized = interp(self.achieved_total_energy_consumption,[0,10000],[0.005,1])
            self.battery_energy_level = self.battery_energy_level - self.achieved_total_energy_consumption
        else:
            self.achieved_total_energy_consumption = 0

        #print(self.battery_energy_level)
        #print('total energy: ', self.achieved_total_energy_consumption)

    def total_processing_delay(self):
        self.achieved_total_processing_delay = self.achieved_local_processing_delay + self.achieved_transmission_delay
        #print('offload ratio: ', self.allocated_offloading_ratio, 'local delay: ', self.achieved_local_processing_delay, 'offlaod delay: ', self.achieved_transmission_delay)
        
    
    def calculate_channel_gain(self):
        #Pathloss gain
        self.pathloss_gain = (math.pow(10,(35.3+37.6*math.log10(self.distance_from_SBS))))/10
        self.small_scale_channel_gain = np.random.rayleigh(1)
        self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        self.total_gain = self.small_scale_channel_gain*self.large_scale_channel_gain#self.pathloss_gain
        if self.total_gain < 0.1:
            self.total_gain = 0.1

    def calculate_assigned_transmit_power_W(self):
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
            energy_efficiency = self.achieved_channel_rate_normalized/self.achieved_total_energy_consumption_normalized #0.4*self.achieved_channel_rate_normalized/0.6*self.achieved_total_energy_consumption_normalized
        
            #energy_efficiency = self.achieved_total_energy_consumption_normalized 
            
        min_energy_efficiency = 0
        max_energy_efficiency = 200
        energy_efficiency = interp(energy_efficiency,[min_energy_efficiency,max_energy_efficiency],[0,1])
        return energy_efficiency
    
    def calculate_throughput_reward(self,communication_channel):
        queue_size = self.user_state_space.calculate_communication_queue_size()
        #normalize queue size
        #queue_size_normalized = interp(queue_size,[0,self.max_communication_qeueu_size],[0,1])
        #print('queue size: ', queue_size)

        #normalize achieved thoughput
        #min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        #achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[min_achievable_rate,max_achievable_rate],[0,1])
        throughput_reward = self.achieved_channel_rate - queue_size
        #print('throughput reward: ', throughput_reward)
        #Normailze throughput reward
        min_throughput_reward = -28960000
        max_throughput_reward = 159844000
        if(throughput_reward > 0):
            throughput_reward_normalized = interp(throughput_reward,[min_throughput_reward,max_throughput_reward],[0,1])
        else:
            throughput_reward_normalized = -0.65
        return throughput_reward_normalized

    def compute_battery_energy_level(self):
        self.battery_energy_level = self.battery_energy_level + self.energy_harvested

    def harvest_energy(self):
        self.energy_harvested = random.randint(0,2000)

    def energy_consumption_reward(self):
        energy_reward = self.battery_energy_level - self.achieved_total_energy_consumption

        max_energy_reward = 22000
        min_energy_reward = 0

        if energy_reward >= 0:
            energy_reward_normalized = interp(energy_reward,[min_energy_reward,max_energy_reward],[0,1])
        else:
            energy_reward_normalized = -0.2

        return energy_reward_normalized

        






  



            


