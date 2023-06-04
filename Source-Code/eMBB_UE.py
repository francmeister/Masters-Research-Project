import pygame, sys, time, random
import random
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
import numpy as np
import math
from State_Space import State_Space
pygame.init()

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,UE_label,screen_position_x,screen_position_y):
        User_Equipment.__init__(self)
        self.eMBB_UE_label = eMBB_UE_label
        self.UE_label = UE_label
        self.eMBB_UE_sprite_width = 87
        self.eMBB_UE_sprite_height = 109
        self.eMBB_UE_screen_position_x = screen_position_x
        self.eMBB_UE_screen_position_y = screen_position_y
        self.filename = 'Resources/eMBB-UE-spritesheet.png'

        #Telecomm Network Properties
        self.max_allowable_latency = 0
        self.max_allowable_reliability = 0
        self.QOS_requirement = QOS_requirement()
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
        self.minislot_counter = 0
        self.x_position = 0
        self.y_position = 0
        self.eMBB_UE_sprite = pygame.image.load(self.filename).convert()
        self.sprite_surface = pygame.Surface((self.eMBB_UE_sprite_width,self.eMBB_UE_sprite_height))
        self.sprite_surface.set_colorkey((0,0,0))
        self.energy_harversted = 0
        self.user_state_space = State_Space(self.UE_label,self.total_gain,self.user_task,self.energy_harversted)
        self.allocated_offloading_ratio = 0
        self.packet_offload_size_bits = 0
        self.packet_local_size_bits = 0
        self.intefering_URLLC_Users = []
        self.offloaded_packet = []

        #self.sprite = SpriteSheet(self.spriteSheetFilename,self.spriteSheet_x,self.spriteSheet_y,self.spriteSheet_width,self.spriteSheet_height)
    def load_eMBB_UE_sprite(self,screen):

        self.sprite_surface.blit(self.eMBB_UE_sprite,(0,0))
        screen.blit(self.sprite_surface,(self.eMBB_UE_screen_position_x,self.eMBB_UE_screen_position_y))

    def generate_task(self,short_TTI,long_TTI):
        self.timeslot_counter+=1

        #After every 1s generate packets with a uniform distribution between 5 and 10 packets per second
        #Long TTI = 0.125 ms. 1 second should be achieved after every 8000 timeslots
        if(self.timeslot_counter*long_TTI >= 1000):
            self.timeslot_counter = 0

            #Require Task Arrival Rate, bits/packets, CPU cycles/packet
            self.task_arrival_rate_packets_per_second = random.randint(5,10) #Packets/s
            self.max_allowable_latency = random.randint(1000,2000) #[1,2] s
            self.max_allowable_reliability = 0
            self.packet_size = (random.randint(50,100))*8000 # [50,100]Kilobytes. 8000 bits in a KB
            self.QOS_requirement.set_requirements(self.max_allowable_latency,self.max_allowable_reliability)
            self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)
            self.communication_queue.append(self.user_task)

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos, Env_width_pixels, Env_width_metres):
        self.x_position = self.eMBB_UE_sprite.get_rect().centerx
        self.y_position = self.eMBB_UE_sprite.get_rect().centery

        x_diff_pixels = abs(SBS_x_pos-self.x_position)
        y_diff_pixels = abs(SBS_y_pos-self.y_position)

        x_diff_metres = (x_diff_pixels/Env_width_pixels)*Env_width_metres
        y_diff_pixels = (y_diff_pixels/Env_width_pixels)*Env_width_metres

        self.distance_from_SBS = math.sqrt(x_diff_metres^2+y_diff_pixels^2)

    def collect_state(self):
        self.user_state_space.collect(self.total_gain,self.communication_queue,self.energy_harversted)
        return self.user_state_space

    def split_packet(self):
        packet_dec = self.communication_queue[0].packet_queue[0]
        packet_bin = bin(packet_dec)[2:]
        packet_size = len(packet_bin)
        self.packet_offload_size_bits = int(self.allocated_offloading_ratio*packet_size)
        self.packet_local_size_bits = int((1-self.allocated_offloading_ratio)*packet_size)
        self.local_queue.append(random.getrandbits(self.packet_local_size_bits))
        self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
        self.dequeue_packet()

    def transmit_to_SBS(self, communication_channel, URLLC_Users):
        #Find URLLC users transmitting on this eMBB user's subcarriers
        subcarrier_URLLC_User_mapping = communication_channel.subcarrier_URLLC_User_mapping_
        for subcarrier in self.allocated_subcarriers:
            self.intefering_URLLC_Users.append(subcarrier_URLLC_User_mapping[subcarrier - 1])

        #Calculate the bandwidth achieved on each subcarrier, each subcarrier receives interference from mapped URLLC users
        achieved_subcarriers_channel_rates = []
        for subcarrier in self.allocated_subcarriers:
            URLLC_users_on_this_subcarrier = self.intefering_URLLC_Users[subcarrier - 1][1]
            URLLC_Users_transmit_powers = []
            URLLC_Users_channel_gains = []
            for URLLC_User in URLLC_users_on_this_subcarrier:
                if URLLC_Users[URLLC_User - 1].has_transmitted_this_time_slot == True:
                    URLLC_Users_transmit_powers.append(URLLC_Users[URLLC_User - 1].assigned_transmit_power_W)
                    URLLC_Users_channel_gains.append(URLLC_Users[URLLC_User - 1].total_gain)
            achieved_subcarrier_channel_rate = self.calculate_channel_gain(URLLC_Users_channel_gains,communication_channel)
            achieved_subcarriers_channel_rates.append(achieved_subcarrier_channel_rate)

        self.achieved_channel_rate = sum(achieved_subcarriers_channel_rates)

        print("allocated_subcarriers", self.allocated_subcarriers)
        print("intefering_URLLC_Users: ", self.intefering_URLLC_Users)
        print(self.intefering_URLLC_Users[0][1])

    def calculate_channel_rate(self,transmitting_URLLC_Users, communication_channel):
        channel_rate = communication_channel.subcarrier_bandwidth_kHz*(1-(len(transmitting_URLLC_Users)/communication_channel.num_minislots_per_timeslot))*math.log2(1+((self.assigned_transmit_power_W*self.total_gain)/(communication_channel.noise_spectral_density_W*communication_channel.subcarrier_bandwidth_kHz*1000)))
        return channel_rate
    
    def local_processing(self):
        cycles_per_packet = self.cpu_cycles_per_byte*(self.packet_size*0.125)
        self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.cpu_clock_frequency,2)*(1-self.allocated_offloading_ratio)*cycles_per_packet
        self.local_queue.pop(0) 

    def offloading(self):
        






  



            


