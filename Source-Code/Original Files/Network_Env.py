import gym
from gym import spaces
import pygame, sys, time, random, numpy as np
from eMBB_UE import eMBB_UE
from Communication_Channel import Communication_Channel
from SBS import SBS
from URLLC_UE import URLLC_UE

pygame.init()

#Set constant variables
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900
ENV_WIDTH_PIXELS = 1100
ENV_HEIGHT_PIXELS = 900
ENV_WIDTH_METRES = 400
ENV_HEIGHT_METRES = 400

clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

class NetworkEnV(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box()
        self.observation_space = spaces.Box()
        self.STEP_LIMIT = 1000
        self.sleep = 0
        self.reset()
        self.create_objects()
        self.steps = 0
       

    def step(self,action):
        reward = 0
        
        #Perform Actions
        self.SBS1.allocate_transmit_powers(self.eMBB_Users,self.URLLC_Users)
        self.SBS1.allocate_offlaoding_ratios(self.eMBB_Users)

        for URLLC_User in self.URLLC_Users:
            URLLC_User.send_packet()

        for eMBB_User in self.eMBB_Users:
            eMBB_User.split_packet()

        for URLLC_User in self.URLLC_Users:
            if URLLC_User.has_transmitted_this_time_slot == True:
                URLLC_User.transmit_to_SBS(self.eMBB_Users, self.Communication_Channel_1)

        for eMBB_User in self.eMBB_Users:
            if eMBB_User.has_transmitted_this_time_slot == True:
                eMBB_User.transmit_to_SBS(self.Communication_Channel_1, self.URLLC_Users)
                eMBB_User.local_processing()
                eMBB_User.offloading()
                eMBB_User.total_energy_consumed()
                eMBB_User.total_processing_delay()

        self.SBS1.count_num_arriving_URLLC_packet(self.URLLC_Users)
        self.SBS1.receive_offload_packets(self.eMBB_Users,self.URLLC_Users)
        self.SBS1.calculate_achieved_total_system_energy_consumption(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_system_processing_delay(self.eMBB_Users)
        self.SBS1.calculate_achieved_total_rate_URLLC_users(self.URLLC_Users)
        self.SBS1.calculate_achieved_total_rate_eMBB_users(self.eMBB_Users)
        self.SBS1.calculate_achieved_URLLC_reliability(self.URLLC_Users)
        self.SBS1.calculate_achieved_system_energy_efficiency()
        reward = self.SBS1.calculate_achieved_system_reward(self.eMBB_Users,self.URLLC_Users)

        #Update game state after performing actions
        for eMBB_User in self.eMBB_Users:
            eMBB_User.calculate_distance_from_SBS(self.SBS1.x_position, self.SBS1.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
            eMBB_User.calculate_channel_gain()
            eMBB_User.generate_task(self.Communication_Channel_1.short_TTI,self.Communication_Channel_1.long_TTI)
            eMBB_User.collect_state()

        for URLLC_User in self.URLLC_Users:
            URLLC_User.calculate_distance_from_SBS(self.SBS1.x_position, self.SBS1.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
            URLLC_User.calculate_channel_gain()
            URLLC_User.generate_task(self.Communication_Channel_1.short_TTI,self.Communication_Channel_1.long_TTI)
            URLLC_User.collect_state()

        observation = self.SBS1.collect_state_space(self.eMBB_Users,self.URLLC_Users)
        done = self.check_timestep()
        info = 0
        self.steps+=1

      
        return observation,reward,done,info
    
    def reset(self):
        self.steps = 0
        self.SBS1.set_properties()
        for eMBB_User in self.eMBB_Users:
            eMBB_User.set_properties_UE()
            eMBB_User.set_properties_eMBB()

        for URLLC_User in self.URLLC_Users:
            URLLC_User.set_properties_UE()
            URLLC_User.set_properties_URLLC()

        self.eMBB_Users.clear()
        self.URLLC_Users.clear()
        self.group_users()

        self.SBS1.associate_users(self.eMBB_Users,self.URLLC_Users)
        self.Communication_Channel_1.set_properties(self.SBS1.SBS_label)

        self.Communication_Channel_1.get_SBS_and_Users(self.SBS1)
        self.Communication_Channel_1.initiate_subcarriers()
        self.Communication_Channel_1.allocate_subcarriers_eMBB(self.eMBB_Users)
        self.Communication_Channel_1.create_resource_blocks_URLLC()
        self.Communication_Channel_1.allocate_resource_blocks_URLLC(self.URLLC_Users)
        self.Communication_Channel_1.subcarrier_URLLC_User_mapping()

        #Plotting timeframe
        for eMBB_User in self.eMBB_Users:
            eMBB_User.set_matplotlib_rectangle_properties(self.Communication_Channel_1.long_TTI)

        for URLLC_User in self.URLLC_Users:
            URLLC_User.set_matplotlib_rectangle_properties(self.Communication_Channel_1)

        #Communication_Channel_1.plot_timeframe(eMBB_Users,URLLC_Users)

        self.SBS1.collect_state_space(self.eMBB_Users,self.URLLC_Users)
        observation = self.SBS1.system_state_space
        return observation

    def render(self, mode='human'):
        pass

    def create_objects(self):
        #Small Cell Base station
        self.SBS1 = SBS(1)

        #Users
        self.URLLC_UE_1 = URLLC_UE(1,2,600,700)
        self.URLLC_UE_2 = URLLC_UE(2,3,600,700)
        self.URLLC_UE_3 = URLLC_UE(3,4,600,700)
        self.URLLC_UE_4 = URLLC_UE(4,5,600,700)
        self.URLLC_UE_5 = URLLC_UE(5,6,600,700)
        self.URLLC_UE_6 = URLLC_UE(6,7,600,700)
        self.URLLC_UE_7 = URLLC_UE(7,8,600,700)
        self.eMBB_UE_1 = eMBB_UE(1,1,100,600)
        self.eMBB_UE_2 = eMBB_UE(2,9,1000,500)
        self.eMBB_UE_3 = eMBB_UE(3,10,1000,500)
        self.eMBB_UE_4 = eMBB_UE(4,11,1000,500)
        self.eMBB_UE_5 = eMBB_UE(5,12,1000,500)
        self.eMBB_UE_6 = eMBB_UE(6,13,1000,500)
        self.eMBB_UE_7 = eMBB_UE(7,14,1000,500)

        #Communication Channel
        self.Communication_Channel_1 = Communication_Channel(self.SBS1.SBS_label)

        #Group Users
        self.eMBB_Users = []
        self.URLLC_Users = []
        self.group_users()

        #Associate SBS with users
        self.SBS1.associate_users(self.eMBB_Users,self.URLLC_Users)


    def group_users(self):
        #Group all eMBB Users
        self.eMBB_Users.append(self.eMBB_UE_1)
        self.eMBB_Users.append(self.eMBB_UE_2)
        self.eMBB_Users.append(self.eMBB_UE_3)
        self.eMBB_Users.append(self.eMBB_UE_4)
        self.eMBB_Users.append(self.eMBB_UE_5)
        self.eMBB_Users.append(self.eMBB_UE_6)
        self.eMBB_Users.append(self.eMBB_UE_7)


        #Group all URLLC users
        self.URLLC_Users.append(self.URLLC_UE_1)
        self.URLLC_Users.append(self.URLLC_UE_2)
        self.URLLC_Users.append(self.URLLC_UE_3)
        self.URLLC_Users.append(self.URLLC_UE_4)
        self.URLLC_Users.append(self.URLLC_UE_5)
        self.URLLC_Users.append(self.URLLC_UE_6)
        self.URLLC_Users.append(self.URLLC_UE_7)

    def check_timestep(self):
        if self.steps >= self.STEP_LIMIT:
            return True