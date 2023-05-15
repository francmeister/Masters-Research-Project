import pygame, sys, time, random
from Spritesheet import SpriteSheet
pygame.init()

class SBS():
    def __init__(self, SBS_label):
        self.SBS_label = SBS_label
        self.spriteSheetFilename = 'Resources/cell-tower-spritesheet.png'
        self.spriteSheet_x = 0
        self.spriteSheet_y = 0
        self.spriteSheet_width = 488.5
        self.spriteSheet_height = 689
        self.sprite = SpriteSheet(self.spriteSheetFilename,self.spriteSheet_x,self.spriteSheet_y,self.spriteSheet_width,self.spriteSheet_height)


        # Set Game objects Positions