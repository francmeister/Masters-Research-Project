import pygame, sys, time, random
from Spritesheet import SpriteSheet
pygame.init()

class URLLC_UE():
    def __init__(self, URLLC_UE_label, screen_position_x,screen_position_y):
        self.URLLC_UE_label = URLLC_UE_label
        self.URLLC_UE_sprite_width = 87
        self.URLLC_UE_sprite_height = 109
        self.URLLC_UE_screen_position_x = screen_position_x
        self.URLLC_UE_screen_position_y = screen_position_y
        self.filename = 'Resources/URLLC-UE-spritesheet.png'

        #Telecomm Network Properties
        self.latency_requirement = 1 # 1 ms
        self.reliability_requirement = 10^-7
        self.achieved_reliability = 0
        self.packet_size = 32 #bytes
        self.task_arrival_rate = 500 #packets/s
        self.achieved_transmission_delay = 0
        
    def load_URLLC_UE_sprite(self,screen):
        URLLC_UE_sprite = pygame.image.load(self.filename).convert()

        sprite_surface = pygame.Surface((self.URLLC_UE_sprite_width,self.URLLC_UE_sprite_height))
        sprite_surface.set_colorkey((0,0,0))

        sprite_surface.blit(URLLC_UE_sprite,(0,0))
        screen.blit(sprite_surface,(self.URLLC_UE_screen_position_x,self.URLLC_UE_screen_position_y))