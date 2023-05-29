import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
import numpy as np
import math
pygame.init()

class User_Equipment():
    def __init__(self):
        #Telecomm Network Properties
        self.max_transmission_power_dBm = 23 # dBm
        self.max_transmission_power_W =  (math.pow(10,(self.max_transmission_power_dBm/10)))/1000# Watts
        self.assigned_transmit_power_dBm = 0
        self.assigned_transmit_power_W = 0
        self.cpu_cycles_per_byte = 330
        self.cpu_clock_frequency = random.randrange(1,2,1)
        self.associated_SBS_label = 0
        self.small_scale_channel_gain = 0
        self.large_scale_channel_gain = 0
        self.pathloss_gain = 0
        self.achieved_channel_rate = 0
        self.achieved_SNIR = 0
        self.task_profile = []
        self.distance_from_SBS = 0
        self.total_gain = 0

    def calculate_channel_gain(self):
        #Pathloss gain
        self.pathloss_gain = (10^(35.3+37.6*math.log10(self.distance_from_SBS)))/10
        self.small_scale_channel_gain = np.random.rayleigh(1)
        self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        self.total_gain = self.small_scale_channel_gain*self.large_scale_channel_gain*self.pathloss_gain

    def calculate_assigned_transmit_power_W(self):
        self.assigned_transmit_power_W = (math.pow(10,(self.assigned_transmit_power_dBm/10)))/1000