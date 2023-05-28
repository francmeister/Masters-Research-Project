import pygame, sys, time, random, numpy as np
from eMBB_UE import eMBB_UE
from Communication_Channel import Communication_Channel
from SBS import SBS
from URLLC_UE import URLLC_UE

pygame.init()

#Set constant variables
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900
ENV_WIDTH = 1100
ENV_HEIGHT = 900

clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

SBS1 = SBS(1)
eMBB_UE_1 = eMBB_UE(1,1,100,600)
URLLC_UE_1 = URLLC_UE(1,2,600,700)
URLLC_UE_2 = URLLC_UE(2,3,600,700)
URLLC_UE_3 = URLLC_UE(3,4,600,700)
URLLC_UE_4 = URLLC_UE(4,5,600,700)
URLLC_UE_5 = URLLC_UE(5,4,600,700)
URLLC_UE_6 = URLLC_UE(6,5,600,700)
URLLC_UE_7 = URLLC_UE(7,5,600,700)
eMBB_UE_2 = eMBB_UE(2,6,1000,500)
Communication_Channel_1 = Communication_Channel()

# Group all eMBB users
eMBB_Users = []
eMBB_Users.append(eMBB_UE_1)
eMBB_Users.append(eMBB_UE_2)

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
#Communication_Channel_1.allocate_resource_blocks_URLLC()


