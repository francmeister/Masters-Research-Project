import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
pygame.init()

class User_Equipment():
    def __init__(self):
        #Telecomm Network Properties
        self.max_transmission_power = 23 # dBm
        self.cpu_cycles_per_byte = 330
        self.cpu_clock_frequency = random.randrange(0.1,1,0.1)
        self.associated_SBS_label = 0
        self.small_scale_channel_gain = 0
        self.large_scale_channel_gain = 0
        self.achieved_channel_rate = 0
        self.achieved_SNIR = 0
        self.task_profile = []