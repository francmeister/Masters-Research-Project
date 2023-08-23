#%%
import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
from eMBB_UE import eMBB_UE
import numpy as np
import matplotlib.pyplot as plt
import math
pygame.init()


class Communication_Channel():
    def __init__(self):
        #Telecomm Network Propertiesset_properties(self,SBS_label):
        self.set_properties()


    def get_SBS_and_Users(self,SBS):
        self.eMBB_Users = SBS.associated_eMBB_users
        self.num_RB_per_eMBB = int(self.num_RB/len(self.eMBB_Users))
        self.num_allocate_RB_lower_bound = self.num_RB_per_eMBB - self.single_side_standard_deviation
        self.num_allocate_RB_upper_bound = self.num_RB_per_eMBB + self.single_side_standard_deviation
        #print("num_subcarriers_per_RB_eMBB: ", self.num_subcarriers_per_RB_eMBB)

    def initiate_RBs(self):
        for i in range(1,self.num_RB + 1):
            self.RBs.append(i)

        for RB in range(1,self.num_RB + 1):
            self.RB_eMBB_mappings.append([RB,0])

    def allocate_RBs_eMBB(self,eMBB_Users,number_of_RBs_action):
        #print("number_of_subcarriers_action",number_of_subcarriers_action)
        upper_bound = self.num_allocate_RB_upper_bound
        number_of_eMBB_Users_left = len(eMBB_Users)
        index = 0
        for eMBB_User in eMBB_Users:
            if self.num_of_available_RBs >= upper_bound:
                #allocate_subcarriers = random.randint(self.num_allocate_subcarriers_lower_bound,self.num_allocate_subcarriers_upper_bound)
                allocate_RBs = number_of_RBs_action[index]
                index+=1
                eMBB_User.number_of_allocated_RBs = allocate_RBs
                eMBB_User.allocated_RBs = self.RBs[0:eMBB_User.number_of_allocated_RBs]
                self.num_of_available_RBs -= allocate_RBs
                self.RBs = self.RBs[eMBB_User.number_of_allocated_RBs:]
                number_of_eMBB_Users_left -= 1

            else:
                allocate_RBs = int(len(self.RBs)/number_of_eMBB_Users_left)
                eMBB_User.number_of_allocated_RBs = allocate_RBs
                eMBB_User.allocated_RBs = self.RBs[0:eMBB_User.number_of_allocated_RBs]
                self.num_of_available_RBs -= allocate_RBs
                self.RBs = self.RBs[eMBB_User.number_of_allocated_RBs:]
                number_of_eMBB_Users_left -= 1

        for eMBB_User in eMBB_Users:
            if eMBB_User.number_of_allocated_RBs > 0:
                for RB in eMBB_User.allocated_RBs:
                    index = self.RB_eMBB_mappings.index([RB,0])
                    self.RB_eMBB_mappings[index] = [RB,eMBB_User.eMBB_UE_label]   

    def set_properties(self):
        self.transmission_queue = []
        self.system_bandwidth_Hz = 120*math.pow(10,6)
        self.subcarrier_bandwidth_Hz = 15*math.pow(10,3) # 15kHz
        self.num_subcarriers_per_RB = 12
        self.RB_bandwidth_Hz = self.subcarrier_bandwidth_Hz*self.num_subcarriers_per_RB
        self.num_RB = int(self.system_bandwidth_Hz/self.RB_bandwidth_Hz)
        self.long_TTI = 0.125 #1ms
        self.noise_spectral_density_dbm = -174 # -174dBM/Hz
        self.noise_spectral_density_W = (math.pow(10,(self.noise_spectral_density_dbm/10)))/1000
        self.single_side_standard_deviation = 5
        self.eMBB_Users = []
        self.RBs = []
        self.RB_eMBB_mappings = []
        self.num_of_available_RBs = self.num_RB
        #self.fig, self.ax = plt.subplots()