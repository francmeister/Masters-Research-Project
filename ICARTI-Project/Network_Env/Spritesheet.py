import pygame

class SpriteSheet:
    def __init__(self,filename,x_coordinate,y_coordinate,width,height):
        self.filename = filename
        self.sprite_sheet = pygame.image.load(filename).convert()
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.width = width
        self.height = height

    def getSprite(self,x_coordinate,y_coordinate,width,height):
        sprite = pygame.Surface((width,height))
        sprite.set_colorkey((0,0,0))
        sprite.blit(self.sprite_sheet,(0,0),(x_coordinate,y_coordinate,width,height))
        return sprite
    
    def animateSpriteSheet(self,frame_counter):
        if frame_counter == 0:
            sprite = self.getSprite(self.x_coordinate,self.y_coordinate,self.width,self.height)
        elif frame_counter == 1:
            sprite = self.getSprite(self.x_coordinate + self.width,self.y_coordinate,self.width,self.height)
        elif frame_counter == 2:
            sprite = self.getSprite(self.x_coordinate + self.width*2,self.y_coordinate,self.width,self.height)
        elif frame_counter == 3:
            self.getSprite(self.x_coordinate + self.width*3,self.y_coordinate,self.width,self.height)

        return sprite

    