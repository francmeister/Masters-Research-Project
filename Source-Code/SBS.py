import pygame, sys, time, random
from Spritesheet import SpriteSheet
pygame.init()

class SBS():
    def __init__(self, SBS_label):
        self.SBS_label = SBS_label
        self.cell_tower_sprite_width = 340
        self.cell_tower_sprite_height = 340
        self.filename1 = 'Resources/cell-tower-spritesheet-1.png'
        self.filename2 = 'Resources/cell-tower-spritesheet-2.png'
        self.filename3 = 'Resources/cell-tower-spritesheet-3.png'
        self.filename4 = 'Resources/cell-tower-spritesheet-4.png'

        #SBS Telecom properties
        self.clock_frequency = 2
        self.work_load = 0
        self.eMBB_UEs = []
        self.URLLC_UEs = []
        self.achieved_users_energy_consumption = 0
        self.achieved_users_channel_rate = 0
        
    def load_cell_tower_sprite(self,screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount):
        cell_tower_sprite1 = pygame.image.load(self.filename1).convert()
        cell_tower_sprite2 = pygame.image.load(self.filename2).convert()
        cell_tower_sprite3 = pygame.image.load(self.filename3).convert()
        cell_tower_sprite4 = pygame.image.load(self.filename4).convert()

        sprite_surface = pygame.Surface((self.cell_tower_sprite_width,self.cell_tower_sprite_height))
        sprite_surface.set_colorkey((0,0,0))

        if frameCount == 0:
            sprite_surface.blit(cell_tower_sprite1,(0,0))
            screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))

        elif frameCount == 1:
            sprite_surface.blit(cell_tower_sprite2,(0,0))
            screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))

        elif frameCount == 2:
            sprite_surface.blit(cell_tower_sprite3,(0,0))
            screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))

        elif frameCount == 3:
            sprite_surface.blit(cell_tower_sprite4,(0,0))
            screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))