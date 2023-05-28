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
        self.number_URLLC_Users_per_RB = 2
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
        self.URLLC_Users = SBS.associated_URLLC_users
        self.num_subcarriers_per_RB_eMBB = int(self.max_num_of_subcarriers/len(self.eMBB_Users))
        self.num_allocate_subcarriers_lower_bound = self.num_subcarriers_per_RB_eMBB - self.single_side_standard_deviation
        self.num_allocate_subcarriers_upper_bound = self.num_subcarriers_per_RB_eMBB + self.single_side_standard_deviation
        print("num_subcarriers_per_RB_eMBB: ", self.num_subcarriers_per_RB_eMBB)

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

        print("eMBB_subcarrier_mappings: ", self.eMBB_subcarrier_mappings)

    def create_resource_blocks_URLLC(self):
        self.number_of_resource_blocks_URLLC = int(len(self.URLLC_Users)/self.number_URLLC_Users_per_RB)
        float_value = len(self.URLLC_Users)/self.number_URLLC_Users_per_RB

        if float_value > self.number_of_resource_blocks_URLLC:
            self.number_of_resource_blocks_URLLC += 1

        self.num_subcarriers_per_RB_URLLC = int(self.max_num_of_subcarriers/self.number_of_resource_blocks_URLLC)
        
        #self.resource_blocks_URLLC = np.arange(1,self.number_of_resource_blocks_URLLC + 1)
        for i in range(1,self.number_of_resource_blocks_URLLC  + 1):
            self.resource_blocks_URLLC.append(i)

        #Map subcarriers to URLLC Resource Block
        start_index = 0
        end_index = self.num_subcarriers_per_RB_URLLC
        subcarriers = []
        for i in range(1,self.max_num_of_subcarriers  + 1):
            subcarriers.append(i)

        for RB in self.resource_blocks_URLLC:
            self.resource_blocks_subcarrier_mappings_URLLC.append(subcarriers[start_index:end_index])
            start_index += self.num_subcarriers_per_RB_URLLC
            end_index += self.num_subcarriers_per_RB_URLLC

        for RB in range(1,self.number_of_resource_blocks_URLLC + 1):
            if self.number_URLLC_Users_per_RB == 1:
                self.resource_blocks_URLLC_mappings.append([RB,0])
            elif self.number_URLLC_Users_per_RB == 2:
                self.resource_blocks_URLLC_mappings.append([RB,0,0])
            elif self.number_URLLC_Users_per_RB == 3:
                self.resource_blocks_URLLC_mappings.append([RB,0,0,0])

        print("number_of_resource_blocks_URLLC: ", self.number_of_resource_blocks_URLLC) 
        print("num_subcarriers_per_RB_URLLC: ", self.num_subcarriers_per_RB_URLLC)     
        print("resource_blocks_URLLC: ", self.resource_blocks_URLLC)
        print("resource_blocks_subcarrier_mappings_URLLC: ", self.resource_blocks_subcarrier_mappings_URLLC)

    def allocate_resource_blocks_URLLC(self,URLLC_Users):
        count1 = 1
        count2 = 0
        for URLLC_User in URLLC_Users:
            URLLC_User.allocated_RB.append(count1)
            count2+=1
            if count2 == self.number_URLLC_Users_per_RB:
                count2 = 0
                count1 += 1

        count1 = 0
        count2 = 0
        prev1 = 0
        prev2 = 0

        for URLLC_User in URLLC_Users:
            if len(URLLC_User.allocated_RB) > 0:
                if self.number_URLLC_Users_per_RB == 1:
                    self.resource_blocks_URLLC_mappings[count1] = [URLLC_User.allocated_RB[0],URLLC_User.URLLC_UE_label]
                    count1 += 1

                if self.number_URLLC_Users_per_RB == 2:
                    if count2 == 0:
                        self.resource_blocks_URLLC_mappings[count1] = [URLLC_User.allocated_RB[0],URLLC_User.URLLC_UE_label,0]
                        count2 += 1
                        prev1 = URLLC_User.URLLC_UE_label
                    elif count2 == 1:
                        self.resource_blocks_URLLC_mappings[count1] = [URLLC_User.allocated_RB[0],prev1,URLLC_User.URLLC_UE_label]
                        count2 += 1
                    if count2 == 2:
                        count2 = 0
                        count1 += 1

                if self.number_URLLC_Users_per_RB == 3:
                    if count2 == 0:
                        self.resource_blocks_URLLC_mappings[count1] = [URLLC_User.allocated_RB[0],URLLC_User.URLLC_UE_label,0,0]
                        count2 += 1
                        prev1 = URLLC_User.URLLC_UE_label
                    elif count2 == 1:
                        self.resource_blocks_URLLC_mappings[count1] = [URLLC_User.allocated_RB[0],prev1,URLLC_User.URLLC_UE_label,0]
                        count2 += 1
                        prev2 = URLLC_User.URLLC_UE_label
                    elif count2 == 2:
                        self.resource_blocks_URLLC_mappings[count1] = [URLLC_User.allocated_RB[0],prev1,prev2,URLLC_User.URLLC_UE_label]
                        count2 += 1
                    if count2 == 3:
                        count2 = 0
                        count1 += 1

        print("resource_blocks_URLLC_mappings: ", self.resource_blocks_URLLC_mappings)
            



        


        



        
    