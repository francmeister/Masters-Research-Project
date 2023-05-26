import pygame, sys, time, random
import random
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
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
        self.max_allowable_latency = 0
        self.max_allowable_reliability = 0
        self.QOS_requirement = QOS_requirement()
        self.packet_size_kilobytes = random.randint(50,100) #Kilobytes
        self.task_arrival_rate_packets_per_second = 0 #Packets/s
        self.user_task = Task(self.cpu_cycles_per_byte)
        self.offloading_ratio = 0
        self.local_computation_delay_seconds = 0
        self.achieved_local_energy_consumption = 0
        self.offload_transmission_energy = 0
        self.battery_energy_level = 100 # Begin with 100%
        self.energy_harvested = 0
        self.achieved_transmission_delay = 0
        self.allocated_subcarriers = []
        self.communication_queue = []
        self.local_queue = []
        self.timeslot_counter = 0
        self.minislot_counter = 0


        #self.sprite = SpriteSheet(self.spriteSheetFilename,self.spriteSheet_x,self.spriteSheet_y,self.spriteSheet_width,self.spriteSheet_height)
    def load_eMBB_UE_sprite(self,screen):
        eMBB_UE_sprite = pygame.image.load(self.filename).convert()

        sprite_surface = pygame.Surface((self.eMBB_UE_sprite_width,self.eMBB_UE_sprite_height))
        sprite_surface.set_colorkey((0,0,0))

        sprite_surface.blit(eMBB_UE_sprite,(0,0))
        screen.blit(sprite_surface,(self.eMBB_UE_screen_position_x,self.eMBB_UE_screen_position_y))

    def generate_task(self,short_TTI,long_TTI):
        self.timeslot_counter+=1

        #After every 1s generate packets with a uniform distribution between 5 and 10 packets per second
        #Long TTI = 0.125 ms. 1 second should be achieved after every 8000 timeslots
        if(self.timeslot_counter*long_TTI >= 1000):
            self.timeslot_counter = 0

            #Require Task Arrival Rate, bits/packets, CPU cycles/packet
            self.task_arrival_rate_packets_per_second = random.randint(5,10) #Packets/s
            self.max_allowable_latency = random.randint(1000,2000) #[1,2] s
            self.max_allowable_reliability = 0
            self.packet_size = (random.randint(50,100))*8000 # [50,100]Kilobytes. 8000 bits in a KB
            self.QOS_requirement.set_requirements(self.max_allowable_latency,self.max_allowable_reliability)
            self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)
            


