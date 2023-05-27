import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
from eMBB_UE import eMBB_UE
pygame.init()

class Communication_Channel():
    def __init__(self,SBS_label):
        #Telecomm Network Properties
        self.transmission_queue = []
        self.max_num_of_subcarriers = 144
        self.subcarrier_bandwidth_kHz = 120 # 120kHz
        self.num_subcarriers_per_RB = 12
        self.SBS_label = SBS_label
        self.eMBB_subcarrier_mappings = []
        self.URLLC_RB_mappings = []
        self.long_TTI = 0.125 #1ms
        self.short_TTI = 0.018 # 0.143ms
        self.num_minislots_per_timeslot = 7
        self.noise_spectral_density = -174 # -174dBM/Hz
        self.resource_block_subcarrier_mapping = []
        self.subcarriers = []

    def initiate_subcarriers(self):
        for i in range(1,self.max_num_of_subcarriers + 1):
            self.subcarriers.append(i)

    def allocate_subcarriers_eMBB(self,eMBB_Users):
        

    def allocate_subcarriers_URLLC(self,URLLC_Users):

    def create_resource_blocks(self):
        number_of_resource_blocks = self.max_num_of_subcarriers/self.num_subcarriers_per_RB
        start_index = 0
        end_index = self.num_subcarriers_per_RB 

        for RB in number_of_resource_blocks:
            self.resource_block_subcarrier_mapping.append(self.subcarriers[start_index:end_index])
            start_index += 12
            end_index += 12

        



        
    