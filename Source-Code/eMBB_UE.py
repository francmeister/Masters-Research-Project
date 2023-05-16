import pygame, sys, time, random
from Spritesheet import SpriteSheet
pygame.init()

class eMBB_UE():
    def __init__(self, eMBB_UE_label,screen_position_x,screen_position_y):
        self.eMBB_UE_label = eMBB_UE_label
        self.eMBB_UE_sprite_width = 87
        self.eMBB_UE_sprite_height = 109
        self.eMBB_UE_screen_position_x = screen_position_x
        self.eMBB_UE_screen_position_y = screen_position_y
        self.filename = 'Resources/eMBB-UE-spritesheet.png'

        #self.sprite = SpriteSheet(self.spriteSheetFilename,self.spriteSheet_x,self.spriteSheet_y,self.spriteSheet_width,self.spriteSheet_height)
    def load_eMBB_UE_sprite(self,screen):
        eMBB_UE_sprite = pygame.image.load(self.filename).convert()

        sprite_surface = pygame.Surface((self.eMBB_UE_sprite_width,self.eMBB_UE_sprite_height))
        sprite_surface.set_colorkey((0,0,0))

        sprite_surface.blit(eMBB_UE_sprite,(0,0))
        screen.blit(sprite_surface,(self.eMBB_UE_screen_position_x,self.eMBB_UE_screen_position_y))
