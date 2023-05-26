import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
from eMBB_UE import eMBB_UE
pygame.init()

class Communication_Channel():
    def __init__(self,SBS_label):
        #Telecomm Network Properties
        self.transmission_queue = []
        self.max_num_of_subcarriers = 128
        self.subcarrier_bandwidth_kHz = 120 # 120kHz
        self.num_subcarriers_per_RB = 12
        self.SBS_label = SBS_label
        self.eMBB_subcarrier_mappings = []
        self.long_TTI = 1 #1ms
        self.short_TTI = 0.143 # 0.143ms
        self.num_minislots_per_timeslot = 7
        self.noise_spectral_density = -174 # -174dBM/Hz
        self.initiate_subcarriers()

    def initiate_subcarriers(self):
        for subcarrier in range(self.max_num_of_subcarriers):
            self.eMBB_subcarrier_mappings.append([subcarrier,0])
            #print("Subcarriers Initialized: ", self.eMBB_subcarrier_mappings[subcarrier])

    def allocate_subcarriers_eMBB(self,eMBB_Users):
        subcarrier = 0
        for eMBB_User in eMBB_Users:
            for x in range(self.num_subcarriers_per_RB):
                eMBB_User.allocated_subcarriers.append(subcarrier)
                subcarrier+=1


        subcarrier = 0
        for eMBB_User in eMBB_Users:
            for x in range(self.num_subcarriers_per_RB):
                self.eMBB_subcarrier_mappings[subcarrier] = [subcarrier,eMBB_User.eMBB_UE_label]
                subcarrier+=1
   
        print(self.eMBB_subcarrier_mappings)

    def allocate_subcarriers_URLLC(self,URLLC_Users):
        



        
    