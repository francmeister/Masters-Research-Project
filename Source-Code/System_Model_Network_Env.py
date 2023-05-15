import pygame, sys, time, random
from SBS import SBS
pygame.init()

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900
cell_tower_sprite_width = 340
cell_tower_sprite_height = 340
clock = pygame.time.Clock()

filename1 = 'Resources/cell-tower-spritesheet-1.png'
filename2 = 'Resources/cell-tower-spritesheet-2.png'
filename3 = 'Resources/cell-tower-spritesheet-3.png'
filename4 = 'Resources/cell-tower-spritesheet-4.png'

screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
run = True
BLACK = pygame.Color(0,0,0)

def load_cell_tower_sprite(filename,width,height):
    filename = filename
    cell_tower_sprite = pygame.image.load(filename).convert()
    sprite_surface = pygame.Surface((width,height))
    sprite_surface.set_colorkey((0,0,0))
    sprite_surface.blit(cell_tower_sprite,(0,0))
    return sprite_surface



sprite_surface_1 = load_cell_tower_sprite(filename1,cell_tower_sprite_width,cell_tower_sprite_height)
sprite_surface_2 = load_cell_tower_sprite(filename2,cell_tower_sprite_width,cell_tower_sprite_height)
sprite_surface_3 = load_cell_tower_sprite(filename3,cell_tower_sprite_width,cell_tower_sprite_height)
sprite_surface_4 = load_cell_tower_sprite(filename4,cell_tower_sprite_width,cell_tower_sprite_height)
frameCount = 0

while run:

    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    if frameCount == 0:
        screen.blit(sprite_surface_1,(SCREEN_WIDTH/2-cell_tower_sprite_width/2,SCREEN_HEIGHT/2-cell_tower_sprite_height))
    elif frameCount == 1:
        screen.blit(sprite_surface_2,(SCREEN_WIDTH/2-cell_tower_sprite_width/2,SCREEN_HEIGHT/2-cell_tower_sprite_height))
    elif frameCount == 2:
        screen.blit(sprite_surface_3,(SCREEN_WIDTH/2-cell_tower_sprite_width/2,SCREEN_HEIGHT/2-cell_tower_sprite_height))
    elif frameCount == 3:
        screen.blit(sprite_surface_4,(SCREEN_WIDTH/2-cell_tower_sprite_width/2,SCREEN_HEIGHT/2-cell_tower_sprite_height))

    pygame.display.update()
    frameCount+=1
    if frameCount == 4:
        frameCount = 0
    print(frameCount)
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
        