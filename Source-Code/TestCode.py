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

#Small Cell Basestation
SBS1 = SBS(1)

#Users
URLLC_UE_1 = URLLC_UE(1,2,600,700)
URLLC_UE_2 = URLLC_UE(2,3,600,700)
URLLC_UE_3 = URLLC_UE(3,4,600,700)
URLLC_UE_4 = URLLC_UE(4,5,600,700)
URLLC_UE_5 = URLLC_UE(5,6,600,700)
URLLC_UE_6 = URLLC_UE(6,7,600,700)
URLLC_UE_7 = URLLC_UE(7,8,600,700)
eMBB_UE_1 = eMBB_UE(1,1,100,600)
eMBB_UE_2 = eMBB_UE(2,9,1000,500)
eMBB_UE_3 = eMBB_UE(3,10,1000,500)
eMBB_UE_4 = eMBB_UE(4,11,1000,500)
eMBB_UE_5 = eMBB_UE(5,12,1000,500)
eMBB_UE_6 = eMBB_UE(6,13,1000,500)
eMBB_UE_7 = eMBB_UE(7,14,1000,500)

#Communication Channel
Communication_Channel_1 = Communication_Channel(SBS1.SBS_label)

#Group all eMBB users
eMBB_Users = []
eMBB_Users.append(eMBB_UE_1)
eMBB_Users.append(eMBB_UE_2)
eMBB_Users.append(eMBB_UE_3)
eMBB_Users.append(eMBB_UE_4)
eMBB_Users.append(eMBB_UE_5)
eMBB_Users.append(eMBB_UE_6)
eMBB_Users.append(eMBB_UE_7)


#Group all URLLC users
URLLC_Users = []
URLLC_Users.append(URLLC_UE_1)
URLLC_Users.append(URLLC_UE_2)
URLLC_Users.append(URLLC_UE_3)
URLLC_Users.append(URLLC_UE_4)
URLLC_Users.append(URLLC_UE_5)
URLLC_Users.append(URLLC_UE_6)
URLLC_Users.append(URLLC_UE_7)

#Associate SBS with users
SBS1.associate_users(eMBB_Users,URLLC_Users)

# Allocate subcarriers to eMBB Users
Communication_Channel_1.get_SBS_and_Users(SBS1)
Communication_Channel_1.initiate_subcarriers()
Communication_Channel_1.allocate_subcarriers_eMBB(eMBB_Users)
Communication_Channel_1.create_resource_blocks_URLLC()
Communication_Channel_1.allocate_resource_blocks_URLLC(URLLC_Users)
Communication_Channel_1.subcarrier_URLLC_User_mapping()

#Plotting timeframe
for eMBB_User in eMBB_Users:
    eMBB_User.set_matplotlib_rectangle_properties(Communication_Channel_1.long_TTI)

for URLLC_User in URLLC_Users:
    URLLC_User.set_matplotlib_rectangle_properties(Communication_Channel_1)

#Communication_Channel_1.plot_timeframe(eMBB_Users,URLLC_Users)

#Simulate timeslots
num_time_slots = np.arange(1,11)
for time_slot in num_time_slots:
    print("Time SLot Number: ", time_slot)
    for eMBB_User in eMBB_Users:
        eMBB_User.calculate_distance_from_SBS(SBS1.x_position, SBS1.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
        eMBB_User.calculate_channel_gain()
        eMBB_User.generate_task(Communication_Channel_1.short_TTI,Communication_Channel_1.long_TTI)
        eMBB_User.collect_state()

    for URLLC_User in URLLC_Users:
        URLLC_User.calculate_distance_from_SBS(SBS1.x_position, SBS1.y_position, ENV_WIDTH_PIXELS, ENV_WIDTH_METRES)
        URLLC_User.calculate_channel_gain()
        URLLC_User.generate_task(Communication_Channel_1.short_TTI,Communication_Channel_1.long_TTI)
        URLLC_User.collect_state()

    SBS1.collect_state_space(eMBB_Users,URLLC_Users)
    SBS1.allocate_transmit_powers(eMBB_Users,URLLC_Users)
    SBS1.allocate_offlaoding_ratios(eMBB_Users)

    for URLLC_User in URLLC_Users:
        URLLC_User.send_packet()

    for eMBB_User in eMBB_Users:
        eMBB_User.split_packet()

    for URLLC_User in URLLC_Users:
        if URLLC_User.has_transmitted_this_time_slot == True:
            URLLC_User.transmit_to_SBS(eMBB_Users, Communication_Channel_1)

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








