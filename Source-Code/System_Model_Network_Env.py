import pygame, sys, time, random

pygame.init()

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900

screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

run = True

while run:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.update()

BLACK = pygame.Color(0,0,0)

class NetworkEnv():
    def __init__(self, frame_size_x,frame_size_y):
        # Set the window Properties
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.env_window = pygame.display.set_mode((frame_size_x,frame_size_y))

    def reset(self):
        self.env_window.fill(BLACK)

        # Set Game objects Positions
        