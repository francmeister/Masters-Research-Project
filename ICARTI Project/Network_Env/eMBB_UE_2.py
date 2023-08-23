import pygame, sys, time, random
import random
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
import numpy as np
from matplotlib.patches import Rectangle
import math
from State_Space import State_Space
pygame.init()

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,x,y):
        User_Equipment.__init__(self)
        self.x_position = x
        self.y_position = y
        self.eMBB_UE_label = eMBB_UE_label
        self.set_properties_eMBB(self.x_position,self.y_position)
 
    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos):
        x_diff = abs(SBS_x_pos-self.x_position)
        y_diff = abs(SBS_y_pos-self.y_position)

        self.distance_from_SBS = math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2))

    def move_user(self,ENV_WIDTH,ENV_HEIGHT,longTTI):
        self.timeslot_counter_2+=1

        if(self.timeslot_counter_2*longTTI > 30000): #Move user every 30 seconds
            self.x_position = random.randint(self.xpos_move_lower_bound,self.xpos_move_upper_bound)
            self.y_position = random.randint(self.ypos_move_lower_bound,self.ypos_move_upper_bound)

            if self.x_position < 0 or self.x_position > ENV_WIDTH:
                self.x_position = self.original_x_pos

            if self.y_position < 0 or self.y_position > ENV_HEIGHT:
                self.y_position = self.original_y_pos

            self.timeslot_counter_2 = 0

    def generate_task(self,long_TTI):
        self.has_transmitted_this_time_slot = False
        self.timeslot_counter+=1

        #After every 1s generate packets with a uniform distribution between 5 and 10 packets per second
        #Long TTI = 0.125 ms. 1 second should be achieved after every 8000 timeslots
        if(self.timeslot_counter*long_TTI >= 1000):
            self.timeslot_counter = 0

            #Require Task Arrival Rate, bits/packets, CPU cycles/packet
            self.task_arrival_rate_packets_per_second = random.randint(5,10) #Packets/s
            self.max_allowable_latency = random.randint(1000,2000) #[1,2] s
            self.max_allowable_reliability = random.randint(1000,3000)
            self.packet_size = (random.randint(50,100))*8000 # [50,100]Kilobytes. 8000 bits in a KB
            self.QOS_requirement.set_requirements(self.max_allowable_latency,self.max_allowable_reliability)
            self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)
            self.communication_queue.append(self.user_task)

    def collect_state(self):
        self.user_state_space.collect(self.total_gain,self.communication_queue,self.energy_harversted,self.QOS_requirement)
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

    def dequeue_packet(self):
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                self.communication_queue[0].packet_queue.pop(0)

            elif len(self.communication_queue[0].packet_queue) == 0:
                self.dequeue_task()

    def dequeue_task(self):
        self.communication_queue.pop(0)

    def transmit_to_SBS(self, communication_channel):
     
        #Calculate the bandwidth achieved on each RB
        achieved_RB_channel_rates = []
        for RB in self.allocated_RBs:
            achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel)
            achieved_RB_channel_rates.append(achieved_RB_channel_rate)

        self.achieved_channel_rate = sum(achieved_RB_channel_rates)

    def calculate_channel_rate(self, communication_channel):
        channel_rate = communication_channel.RB_bandwidth_Hz*math.log2(1+((self.assigned_transmit_power_W*self.total_gain)/(communication_channel.noise_spectral_density_W*communication_channel.RB_bandwidth_Hz)))
        return channel_rate
    
    def local_processing(self):
        cycles_per_packet = self.cpu_cycles_per_byte*(self.packet_size*0.125)
        self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.cpu_clock_frequency,2)*(1-self.allocated_offloading_ratio)*cycles_per_packet
        self.achieved_local_processing_delay = ((1-self.allocated_offloading_ratio)*cycles_per_packet)/self.cpu_clock_frequency
        self.local_queue.pop(0) 

    def offloading(self):
        self.achieved_transmission_delay = self.packet_offload_size_bits/self.achieved_channel_rate
        self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*self.achieved_transmission_delay

    def total_energy_consumed(self):
        self.achieved_total_energy_consumption = self.achieved_local_energy_consumption + self.achieved_transmission_energy_consumption

    def total_processing_delay(self):
        self.achieved_total_processing_delay = self.achieved_local_processing_delay + self.achieved_transmission_delay


    def set_properties_eMBB(self,x,y):
        self.max_allowable_latency = 2000 #[1,2] s
        self.min_allowable_latency = 1000 #[1,2] s
        self.max_allowable_reliability = 3
        self.min_allowable_reliability = 0
        self.QOS_requirement = QOS_requirement()
        self.QOS_requirement_for_transmission = QOS_requirement()
        self.packet_size_kilobytes = random.randint(50,100) #Kilobytes
        self.task_arrival_rate_packets_per_second = 0 #Packets/s
        self.user_task = Task(330)
        self.offloading_ratio = 0
        self.local_computation_delay_seconds = 0
        self.achieved_local_energy_consumption = 0
        self.offload_transmission_energy = 0
        self.battery_energy_level = 100 # Begin with 100%
        self.energy_harvested = 0
        self.achieved_transmission_delay = 0
        self.allocated_subcarriers = []
        self.number_of_allocated_subcarriers = 0
        self.local_queue = []
        self.timeslot_counter = 0
        self.timeslot_counter_2 = 0
        self.minislot_counter = 0
        self.x_position = 0
        self.y_position = 0
        self.energy_harversted = 0
        self.user_state_space = State_Space(self.eMBB_UE_label,self.total_gain,self.communication_queue,self.energy_harversted,self.QOS_requirement)
        self.allocated_offloading_ratio = 0
        self.packet_offload_size_bits = 0
        self.packet_local_size_bits = 0
        self.packet_size = 0
        self.intefering_URLLC_Users = []
        self.offloaded_packet = []
        self.single_side_standard_deviation_pos = 5
        self.x_position = x
        self.y_position = y
        self.original_x_pos = x
        self.original_y_pos = y
        self.xpos_move_lower_bound = self.x_position - self.single_side_standard_deviation_pos
        self.xpos_move_upper_bound = self.x_position + self.single_side_standard_deviation_pos
        self.ypos_move_lower_bound = self.y_position - self.single_side_standard_deviation_pos
        self.ypos_move_upper_bound = self.y_position + self.single_side_standard_deviation_pos
        self.rectangles = []
        #self.r,self.g,self.b = self.random_color_generator()
        self.min_communication_qeueu_size = 0
        self.max_communication_qeueu_size = 8000000
       
