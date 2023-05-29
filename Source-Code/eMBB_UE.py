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
        self.communication_queue = []
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
        self.user_state_space.collect(self.total_gain,self.user_task,self.energy_harversted)
        return self.user_state_space
    
    def dequeue_packet(self):
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                self.communication_queue[0].packet_queue.pop(0)

            elif len(self.communication_queue[0].packet_queue) == 0:
                self.dequeue_task()

    def dequeue_task(self):
        self.communication_queue.pop(0)

    def split_packet(self):
        packet_dec = self.communication_queue[0].packet_queue[0]
        packet_bin = bin(packet_dec)[2:]
        packet_size = len(packet_bin)
        self.packet_offload_size_bits = int(self.allocated_offloading_ratio*packet_size)
        self.packet_local_size_bits = int((1-self.allocated_offloading_ratio)*packet_size)
        self.local_queue.append(random.getrandbits(self.packet_local_size_bits))
        self.dequeue_packet()

    def transmit_to_SBS(self, subcarrier_URLLC_User_mapping):
        for subcarrier in self.allocated_subcarriers:
            self.intefering_URLLC_Users.append(subcarrier_URLLC_User_mapping[subcarrier - 1])

        print("allocated_subcarriers", self.allocated_subcarriers)
        print("intefering_URLLC_Users: ", self.intefering_URLLC_Users)




  



            


