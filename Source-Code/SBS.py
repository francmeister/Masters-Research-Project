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
        self.x_position = 0
        self.y_position = 0

        self.cell_tower_sprite1 = pygame.image.load(self.filename1).convert()
        self.cell_tower_sprite2 = pygame.image.load(self.filename2).convert()
        self.cell_tower_sprite3 = pygame.image.load(self.filename3).convert()
        self.cell_tower_sprite4 = pygame.image.load(self.filename4).convert()

        self.sprite_surface = pygame.Surface((self.cell_tower_sprite_width,self.cell_tower_sprite_height))
        self.sprite_surface.set_colorkey((0,0,0))
        self.associated_users = []
        self.associated_URLLC_users = []
        self.associated_eMBB_users = []
        self.system_state_space = []
        
    def load_cell_tower_sprite(self,screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount):
        if frameCount == 0:
            self.sprite_surface.blit(self.cell_tower_sprite1,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            screen.blit(self.sprite_surface,(400,0))

        elif frameCount == 1:
            self.sprite_surface.blit(self.cell_tower_sprite2,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            screen.blit(self.sprite_surface,(400,0))


        elif frameCount == 2:
            self.sprite_surface.blit(self.cell_tower_sprite3,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            screen.blit(self.sprite_surface,(400,0))

        elif frameCount == 3:
            self.sprite_surface.blit(self.cell_tower_sprite4,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            screen.blit(self.sprite_surface,(400,0))

    def associate_users(self, eMBB_Users, URLLC_Users):
        self.associated_eMBB_users = eMBB_Users
        self.associated_URLLC_users = URLLC_Users

    def get_SBS_center_pos(self):
        self.x_position = self.cell_tower_sprite1.get_rect().centerx
        self.y_position = self.cell_tower_sprite1.get_rect().centery

    def collect_state_space(self, system_users):
        self.associated_users = system_users

        self.system_state_space.clear()
        for user in system_users:
            self.system_state_space.append(user.collect_state())
