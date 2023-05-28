import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
from eMBB_UE import eMBB_UE
import numpy as np
pygame.init()

class Communication_Channel():
    def __init__(self):
        #Telecomm Network Properties
        self.transmission_queue = []
        self.max_num_of_subcarriers = 144
        self.subcarrier_bandwidth_kHz = 120 # 120kHz
        self.num_subcarriers_per_RB_eMBB = 0
        self.num_subcarriers_per_RB_URLLC = 0
        self.number_URLLC_Users_per_RB = 3
        self.number_of_resource_blocks_URLLC = 0
        self.SBS_label = 0
        self.eMBB_subcarrier_mappings = []
        self.URLLC_RB_mappings = []
        self.long_TTI = 0.125 #1ms
        self.short_TTI = 0.018 # 0.143ms
        self.num_minislots_per_timeslot = 7
        self.noise_spectral_density = -174 # -174dBM/Hz
        self.resource_block_subcarrier_mapping_eMBB = []
        self.subcarriers = []
        self.num_of_available_subcarriers = self.max_num_of_subcarriers
        self.single_side_standard_deviation = 5
        self.num_allocate_subcarriers_lower_bound = self.num_subcarriers_per_RB_eMBB - self.single_side_standard_deviation
        self.num_allocate_subcarriers_upper_bound = self.num_subcarriers_per_RB_eMBB + self.single_side_standard_deviation
        self.eMBB_Users = []
        self.URLLC_Users = []
        self.resource_blocks_URLLC = []
        self.resource_blocks_subcarrier_mappings_URLLC = []
        self.resource_blocks_URLLC_mappings = []

    def get_SBS_and_Users(self,SBS):
        self.SBS_label = SBS.SBS_label
        self.eMBB_Users = SBS.associated_eMBB_users
        self.URLLC_Users = SBS.associated_eMBB_users
        self.num_subcarriers_per_RB = int(self.max_num_of_subcarriers/len(self.eMBB_Users))
        self.num_allocate_subcarriers_lower_bound = self.num_subcarriers_per_RB_eMBB - self.single_side_standard_deviation
        self.num_allocate_subcarriers_upper_bound = self.num_subcarriers_per_RB_eMBB + self.single_side_standard_deviation

    def initiate_subcarriers(self):
        for i in range(1,self.max_num_of_subcarriers + 1):
            self.subcarriers.append(i)

        for subcarrier in range(1,self.max_num_of_subcarriers + 1):
            self.eMBB_subcarrier_mappings.append([subcarrier,0])

    def allocate_subcarriers_eMBB(self,eMBB_Users):
        upper_bound = self.num_allocate_subcarriers_upper_bound
        number_of_eMBB_Users_left = len(eMBB_Users)
        for eMBB_User in eMBB_Users:
            if self.num_of_available_subcarriers >= upper_bound:
                allocate_subcarriers = random.randint(self.num_allocate_subcarriers_lower_bound,self.num_allocate_subcarriers_upper_bound)
                eMBB_User.number_of_allocated_subcarriers = allocate_subcarriers
                eMBB_User.allocated_subcarriers = self.subcarriers[0:eMBB_User.number_of_allocated_subcarriers]
                self.num_of_available_subcarriers -= allocate_subcarriers
                self.subcarriers = self.subcarriers[eMBB_User.number_of_allocated_subcarriers:]
                number_of_eMBB_Users_left -= 1

            else:
                allocate_subcarriers = int(len(self.subcarriers)/number_of_eMBB_Users_left)
                eMBB_User.number_of_allocated_subcarriers = allocate_subcarriers
                eMBB_User.allocated_subcarriers = self.subcarriers[0:eMBB_User.number_of_allocated_subcarriers]
                self.num_of_available_subcarriers -= allocate_subcarriers
                self.subcarriers = self.subcarriers[eMBB_User.number_of_allocated_subcarriers:]
                number_of_eMBB_Users_left -= 1

        for eMBB_User in eMBB_Users:
            if eMBB_User.number_of_allocated_subcarriers > 0:
                for subcarrier in eMBB_User.allocated_subcarriers:
                    index = self.eMBB_subcarrier_mappings.index([subcarrier,0])
                    self.eMBB_subcarrier_mappings[index] = [subcarrier,eMBB_User.eMBB_UE_label]

    def allocate_resource_blocks_URLLC(self,URLLC_Users):

    def create_resource_blocks(self):
        self.number_of_resource_blocks_URLLC = len(self.URLLC_Users)/self.number_URLLC_Users_per_RB
        self.num_subcarriers_per_RB_URLLC = self.max_num_of_subcarriers/self.number_of_resource_blocks_URLLC
        
        self.resource_blocks_URLLC.append(np.arange(1,self.number_of_resource_blocks_URLLC))

        #Map subcarriers to URLLC Resource Block
        start_index = 0
        end_index = self.num_subcarriers_per_RB_eMBB

        for RB in self.resource_blocks_URLLC:
            self.reso


        



        
    