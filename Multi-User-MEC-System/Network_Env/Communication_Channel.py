#%%
import pygame, sys, time, random
import random
#from eMBB_UE import eMBB_UE
import numpy as np
import matplotlib.pyplot as plt
import math
pygame.init()

class Communication_Channel():
    def __init__(self,SBS_label):
        #Telecomm Network Propertiesset_properties(self,SBS_label):
        self.set_properties()
        
    def get_SBS_and_Users(self,SBS):
        self.eMBB_Users = SBS.associated_eMBB_users
        self.URLLC_Users = SBS.associated_URLLC_users
        self.num_of_RBs_per_User = self.num_allocate_RBs_upper_bound/len(self.eMBB_Users)
        if len(self.URLLC_Users) > 0:
            self.num_urllc_users_per_RB = self.num_allocate_RBs_upper_bound/len(self.URLLC_Users)
        else:
            self.num_urllc_users_per_RB = 0
        #self.num_RB_per_eMBB = int(self.num_RB/len(self.eMBB_Users))
        #self.num_allocate_RB_lower_bound = self.num_RB_per_eMBB - self.single_side_standard_deviation
        #self.num_allocate_RB_upper_bound = self.num_RB_per_eMBB + self.single_side_standard_deviation
        #print("num_subcarriers_per_RB_eMBB: ", self.num_subcarriers_per_RB_eMBB)

    def initiate_RBs(self):
        for i in range(1,self.num_RB + 1):
            self.RBs.append(i)

        for RB in range(1,self.num_RB + 1):
            self.RB_eMBB_mappings.append([RB,0])

    def allocate_RBs_eMBB(self,eMBB_Users,RB_allocation):
        #self.number_of_RBs_available = self.num_allocate_RBs_upper_bound

        count = 0
        for eMBB_User in eMBB_Users:
            eMBB_User.allocated_RBs = []
            eMBB_User.allocated_RBs = RB_allocation[count]
            count+=1
        '''
        index = 0
        for eMBB_User in eMBB_Users:
            eMBB_User.allocated_RBs.clear()
            for i in range(1,number_of_RBs_action[index]+1):
                if self.number_of_RBs_available > 0:
                    eMBB_User.allocated_RBs.append(i) 
                    self.number_of_RBs_available -= 1
            index+=1
        self.allocated_RBs.clear()
        for eMBB_User in eMBB_Users:
            #print('eMBB: ', eMBB_User.UE_label)
            #print('Number of allocated RBs: ', len(eMBB_User.allocated_RBs))
            self.allocated_RBs.append(len(eMBB_User.allocated_RBs))

        #print('Allocated RBs: ', self.allocated_RBs)
        '''

    def set_properties(self):
        self.subcarrier_bandwidth_Hz = 15*math.pow(10,3) # 15kHz
        #self.subcarrier_bandwidth_Hz = 60*math.pow(10,3)
        self.num_subcarriers_per_RB = 12
        #self.num_subcarriers_per_RB = 20
        self.RB_bandwidth_Hz = self.subcarrier_bandwidth_Hz*self.num_subcarriers_per_RB
        self.num_RB = 55#int(self.system_bandwidth_Hz/self.RB_bandwidth_Hz)
        self.system_bandwidth_Hz = self.RB_bandwidth_Hz*self.num_RB*2
        self.long_TTI = 1 #1ms
        self.noise_spectral_density_dbm = -174 # -174dBM/Hz
        self.noise_spectral_density_W = (math.pow(10,(self.noise_spectral_density_dbm/10)))/1000
        #print('self.noise_spectral_density_W: ', self.noise_spectral_density_W)
        self.single_side_standard_deviation = 5
        self.eMBB_Users = []
        self.RBs = []
        self.RB_eMBB_mappings = []
        self.num_of_available_RBs = self.num_RB
        self.num_allocate_RBs_upper_bound = 55
        self.num_allocate_RBs_lower_bound = 1
        self.number_of_RBs_available = self.num_allocate_RBs_upper_bound
        self.allocated_RBs = []
        self.num_of_RBs_per_User = 0
        self.time_divisions_per_slot = 2
        self.num_of_mini_slots = 7
        #self.fig, self.ax = plt.subplots()




            



        


        



        
    
# %%
