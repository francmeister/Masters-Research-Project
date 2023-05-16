import pygame, sys, time, random
from SBS import SBS
from eMBB_UE import eMBB_UE
from URLLC_UE import URLLC_UE
pygame.init()

#Set constant variables
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900
clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
run = True
BLACK = pygame.Color(0,0,0)
frameCount = 0

#Instantiate objects
SBS1 = SBS(1)
eMBB_UE_1 = eMBB_UE(1,100,600)
URLLC_UE_1 = URLLC_UE(1,600,700)
eMBB_UE_2 = eMBB_UE(1,1100,500)

while run:

    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    #Animate the SBS sprite
    if frameCount == 0:
        SBS1.load_cell_tower_sprite(screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount)
    elif frameCount == 1:
        SBS1.load_cell_tower_sprite(screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount)
    elif frameCount == 2:
        SBS1.load_cell_tower_sprite(screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount)
    elif frameCount == 3:
        SBS1.load_cell_tower_sprite(screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount)

    #load and display the different users' sprites
    eMBB_UE_1.load_eMBB_UE_sprite(screen)
    eMBB_UE_2.load_eMBB_UE_sprite(screen)
    URLLC_UE_1.load_URLLC_UE_sprite(screen)

    pygame.display.update()
    frameCount+=1
    if frameCount == 4:
        frameCount = 0
    clock.tick(10)

class NetworkEnv():
    def __init__(self, frame_size_x,frame_size_y):
        # Set the window Properties
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.env_window = pygame.display.set_mode((frame_size_x,frame_size_y))

    def reset(self):
        self.env_window.fill(BLACK)

        # Set Game objects Positions
        