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
eMBB_UE_2 = eMBB_UE(2,3,1000,500)
Communication_Channel_1 = Communication_Channel()

# Group all eMBB users
eMBB_Users = []
eMBB_Users.append(eMBB_UE_1)
eMBB_Users.append(eMBB_UE_2)

#Group all URLLC users
URLLC_Users = []
URLLC_Users.append(URLLC_UE_1)

#Associate SBS with users
SBS1.associate_users(eMBB_Users,URLLC_Users)

# Allocate subcarriers to eMBB Users
Communication_Channel_1.get_SBS_and_Users(SBS1)
Communication_Channel_1.initiate_subcarriers()
Communication_Channel_1.allocate_subcarriers_eMBB(eMBB_Users)

print("eMBB 1: ",eMBB_UE_1.allocated_subcarriers)
print("eMBB 2: ",eMBB_UE_2.allocated_subcarriers)

print(Communication_Channel_1.eMBB_subcarrier_mappings)
