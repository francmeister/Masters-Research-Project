import pygame, sys, time, random
from Spritesheet import SpriteSheet
import random
from User_Equipment import User_Equipment
pygame.init()

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,screen_position_x,screen_position_y):
        self.eMBB_UE_label = eMBB_UE_label
        self.eMBB_UE_sprite_width = 87
        self.eMBB_UE_sprite_height = 109
        self.eMBB_UE_screen_position_x = screen_position_x
        self.eMBB_UE_screen_position_y = screen_position_y
        self.filename = 'Resources/eMBB-UE-spritesheet.png'

        #Telecomm Network Properties
        self.delay_requirement_seconds = random.randint(8,9) #[8,9] s
        self.packet_size_kilobytes = random.randint(50,100) #Kilobytes
        self.task_arrival_rate_packets_per_second = random.randint(5,10) #Packets/s
        self.offloading_ratio = 0
        self.local_computation_delay_seconds = 0
        self.achieved_local_energy_consumption = 0
        self.offload_transmission_energy = 0
        self.battery_energy_level = 100 # Begin with 100%
        self.energy_harvested = 0
        self.achieved_transmission_delay = 0
        self.allocated_subcarriers = []


        #self.sprite = SpriteSheet(self.spriteSheetFilename,self.spriteSheet_x,self.spriteSheet_y,self.spriteSheet_width,self.spriteSheet_height)
    def load_eMBB_UE_sprite(self,screen):
        eMBB_UE_sprite = pygame.image.load(self.filename).convert()

        sprite_surface = pygame.Surface((self.eMBB_UE_sprite_width,self.eMBB_UE_sprite_height))
        sprite_surface.set_colorkey((0,0,0))

        sprite_surface.blit(eMBB_UE_sprite,(0,0))
        screen.blit(sprite_surface,(self.eMBB_UE_screen_position_x,self.eMBB_UE_screen_position_y))

