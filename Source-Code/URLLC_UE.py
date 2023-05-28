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
        self.packet_size = 32 #bytes
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
                self.packet_size = 32*8 # 32 bytes. 8 bits in a byte
                self.QOS_requirement.set_requirements(self.max_allowable_latency,self.max_allowable_reliability)
                self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)

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
