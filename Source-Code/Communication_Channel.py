import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
pygame.init()

class Communication_Channel():
    def __init__(self,SBS_label):
        #Telecomm Network Properties
        self.transmission_queue = []
        self.max_num_of_subcarriers = 128
        self.subcarrier_bandwidth = 120 # 120kHz
        self.num_subcarriers_per_RB = 12
        self.SBS_label = SBS_label
        self.eMBB_RB_mappings = []
        self.long_TTI = 1 #1ms
        self.short_TTI = 0.143 # 0.143ms
        self.num_minislots_per_timeslot = 7
        self.noise_spectral_density = -174 # -174dBM/Hz