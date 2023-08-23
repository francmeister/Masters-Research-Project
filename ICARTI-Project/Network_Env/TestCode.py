import pygame, sys, time, random, numpy as np
from eMBB_UE_2 import eMBB_UE_2
from Communication_channel_2 import Communication_Channel_2
from SBS import SBS

pygame.init()

#Set constant variables
ENV_WIDTH = 400 #400m
ENV_HEIGHT = 400 #400m


clock = pygame.time.Clock()
#screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

#Small Cell Basestation
SBS1 = SBS(1,ENV_WIDTH/2,ENV_HEIGHT/2)

#Users
'''
eMBB_UE_1 = eMBB_UE_2(1)
eMBB_UE_2_ = eMBB_UE_2(2)
eMBB_UE_3 = eMBB_UE_2(3)
eMBB_UE_4 = eMBB_UE_2(4)
eMBB_UE_5 = eMBB_UE_2(5)
eMBB_UE_6 = eMBB_UE_2(6)
eMBB_UE_7 = eMBB_UE_2(7)
'''

#Communication Channel
Communication_Channel_1 = Communication_Channel_2()

#Group all eMBB users
eMBB_Users = []
number_of_users = 7
for i in range(1,number_of_users+1):
    x_pos = random.randint(0,ENV_WIDTH)
    y_pos = random.randint(0,ENV_HEIGHT)
    eMBB_Users.append(eMBB_UE_2(i,x_pos,y_pos))
'''
eMBB_Users.append(eMBB_UE_1)
eMBB_Users.append(eMBB_UE_2_)
eMBB_Users.append(eMBB_UE_3)
eMBB_Users.append(eMBB_UE_4)
eMBB_Users.append(eMBB_UE_5)
eMBB_Users.append(eMBB_UE_6)
eMBB_Users.append(eMBB_UE_7)
'''



#Associate SBS with users
SBS1.associate_users(eMBB_Users)

# Allocate subcarriers to eMBB Users
Communication_Channel_1.get_SBS_and_Users(SBS1)
Communication_Channel_1.initiate_RBs()
Communication_Channel_1.allocate_RBs_eMBB(eMBB_Users)

#Communication_Channel_1.plot_timeframe(eMBB_Users,URLLC_Users)

#Simulate timeslots
num_time_slots = np.arange(1,11)
for time_slot in num_time_slots:
    print("Time SLot Number: ", time_slot)
    for eMBB_User in eMBB_Users:
        eMBB_User.calculate_distance_from_SBS(SBS1.x_position, SBS1.y_position)
        eMBB_User.calculate_channel_gain()
        eMBB_User.generate_task(Communication_Channel_1.long_TTI)
        eMBB_User.collect_state()

    SBS1.collect_state_space(eMBB_Users)
    SBS1.allocate_transmit_powers(eMBB_Users)
    SBS1.allocate_offlaoding_ratios(eMBB_Users)

    for eMBB_User in eMBB_Users:
        eMBB_User.split_packet()

    for eMBB_User in eMBB_Users:
        if eMBB_User.has_transmitted_this_time_slot == True:
            eMBB_User.transmit_to_SBS(Communication_Channel_1, URLLC_Users)
            eMBB_User.local_processing()
            eMBB_User.offloading()
            eMBB_User.total_energy_consumed()
            eMBB_User.total_processing_delay()

    SBS1.count_num_arriving_URLLC_packet(URLLC_Users)
    SBS1.receive_offload_packets(eMBB_Users,URLLC_Users)
    SBS1.calculate_achieved_total_system_energy_consumption(eMBB_Users)
    SBS1.calculate_achieved_total_system_processing_delay(eMBB_Users)
    SBS1.calculate_achieved_total_rate_URLLC_users(URLLC_Users)
    SBS1.calculate_achieved_total_rate_eMBB_users(eMBB_Users)
    SBS1.calculate_achieved_URLLC_reliability(URLLC_Users)
    SBS1.calculate_achieved_system_energy_efficiency()
    SBS1.calculate_achieved_system_reward(eMBB_Users,URLLC_Users)








