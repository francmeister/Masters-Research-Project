import pygame, sys, time, random, numpy as np
from Spritesheet import SpriteSheet
from QOS_requirement import QOS_requirement
from Task import Task
from User_Equipment import User_Equipment
from State_Space import State_Space
import math
pygame.init()

class URLLC_UE(User_Equipment):
    def __init__(self, URLLC_UE_label,UE_label ,screen_position_x,screen_position_y):
        User_Equipment.__init__(self)
        self.URLLC_UE_label = URLLC_UE_label
        self.UE_label = UE_label
        self.URLLC_UE_sprite_width = 87
        self.URLLC_UE_sprite_height = 109
        self.URLLC_UE_screen_position_x = screen_position_x
        self.URLLC_UE_screen_position_y = screen_position_y
        self.filename = 'Resources/URLLC-UE-spritesheet.png'

        #Telecomm Network Properties
        self.max_allowable_latency = 0
        self.max_allowable_reliability = 0
        self.QOS_requirement = QOS_requirement()
        self.achieved_reliability = 0
        self.packet_size_bytes = 32 #bytes
        self.packet_size_bits = self.packet_size_bytes*8
        self.task_arrival_rate = 500 #packets/s
        self.achieved_transmission_delay = 0
        self.timeslot_counter = 0
        self.minislot_counter = 0
        self.minislot_label = 0
        self.user_task = Task(330)
        self.URLLC_UE_sprite = pygame.image.load(self.filename).convert()
        self.sprite_surface = pygame.Surface((self.URLLC_UE_sprite_width,self.URLLC_UE_sprite_height))
        self.sprite_surface.set_colorkey((0,0,0))
        self.energy_harversted = 0
        self.user_state_space = State_Space(self.UE_label,self.total_gain,self.user_task,self.energy_harversted)
        self.allocated_RB = []
        self.packet_offload_size_bits = 0
        self.offloaded_packet = 0
        
    def load_URLLC_UE_sprite(self,screen):
        self.sprite_surface.blit(self.URLLC_UE_sprite,(0,0))
        screen.blit(self.sprite_surface,(self.URLLC_UE_screen_position_x,self.URLLC_UE_screen_position_y))

    def generate_task(self,short_TTI,long_TTI):
        self.timeslot_counter+=1
        number_of_trials = 1
        probability = 0.5
        sample_size = 1

        #After every 1s generate packets with a bernoulli distribution between 500 packets per second
        #We use the binomial dsitribution which indirectly performs the bernoulli distribution by assigning the sample size to 1 trial
        #Long TTI = 0.125 ms. 1 second should be achieved after every 8000 timeslots
        if(self.timeslot_counter*long_TTI >= 1000):
            self.timeslot_counter = 0
            #Require Task Arrival Rate, bits/packets, CPU cycles/packet
            x = np.random.binomial(number_of_trials,probability,sample_size)

            if(x == 1):
                self.task_arrival_rate_packets_per_second = 500 #Packets/s
                self.max_allowable_latency = 1 #1 ms
                self.max_allowable_reliability = 10^-7
                self.packet_size_bits = 32*8 # 32 bytes. 8 bits in a byte
                self.QOS_requirement.set_requirements(self.max_allowable_latency,self.max_allowable_reliability)
                self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)
                self.communication_queue.append(self.user_task)

    def set_minislot_label(self,minislot_label):
        self.minislot_label = minislot_label

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos, Env_width_pixels, Env_width_metres):
        self.x_position = self.URLLC_UE_sprite.get_rect().centerx
        self.y_position = self.URLLC_UE_sprite.get_rect().centery

        x_diff_pixels = abs(SBS_x_pos-self.x_position)
        y_diff_pixels = abs(SBS_y_pos-self.y_position)

        x_diff_metres = (x_diff_pixels/Env_width_pixels)*Env_width_metres
        y_diff_pixels = (y_diff_pixels/Env_width_pixels)*Env_width_metres

        self.distance_from_SBS = math.sqrt(x_diff_metres^2+y_diff_pixels^2)

    def collect_state(self):
        self.user_state_space.collect(self.total_gain,self.user_task,self.energy_harversted)
        return self.user_state_space
    
    def transmit_to_SBS(self, eMBB_Users,communication_channel):
        print("eMBB users Subcarrier Mapiings", communication_channel.eMBB_subcarrier_mappings)
        subcarrier_eMBB_User_mapping = communication_channel.eMBB_subcarrier_mappings

        achieved_subcarriers_channel_rates = []
        for RB in self.allocated_RB:
            allocated_subcarriers = communication_channel.resource_blocks_subcarrier_mappings_URLLC[RB - 1]
            for subcarrier in allocated_subcarriers:
                interfering_eMBB_user = communication_channel.eMBB_subcarrier_mappings[subcarrier - 1][1]
                interfering_eMBB_user_transmit_power = eMBB_Users[interfering_eMBB_user - 1].assigned_transmit_power_W
                interfering_eMBB_user_channel_gain = eMBB_Users[interfering_eMBB_user - 1].total_gain
                achieved_subcarrier_channel_gain = self.calculate_channel_rate(interfering_eMBB_user_transmit_power,interfering_eMBB_user_channel_gain,communication_channel)
                achieved_subcarriers_channel_rates.append(achieved_subcarrier_channel_gain)
        
        self.achieved_channel_rate = sum(achieved_subcarriers_channel_rates)


        print("allocated RBs", self.allocated_RB)
        print("resource_blocks_subcarrier_mappings_URLLC",communication_channel.resource_blocks_subcarrier_mappings_URLLC)

    def calculate_channel_rate(self,interfering_eMBB_user_transmit_power,achieved_subcarrier_channel_gain,communication_channel):
            channel_rate = communication_channel.subcarrier_bandwidth_kHz*math.log2(1+((self.assigned_transmit_power_W*self.total_gain)/(communication_channel.noise_spectral_density_W*communication_channel.subcarrier_bandwidth_kHz*1000) + interfering_eMBB_user_transmit_power*achieved_subcarrier_channel_gain))
            return channel_rate
    
    def send_packet(self):
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                packet_dec = self.communication_queue[0].packet_queue[0]
                packet_bin = bin(packet_dec)[2:]
                self.packet_offload_size_bits = len(packet_bin)
                self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
                self.dequeue_packet()
