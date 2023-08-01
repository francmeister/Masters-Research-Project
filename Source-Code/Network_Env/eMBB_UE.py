import pygame, sys, time, random
import random
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
import numpy as np
from matplotlib.patches import Rectangle
import math
from State_Space import State_Space
pygame.init()

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,UE_label,screen_position_x,screen_position_y):
        User_Equipment.__init__(self)
        self.eMBB_UE_label = eMBB_UE_label
        self.UE_label = UE_label
        self.eMBB_UE_sprite_width = 87
        self.eMBB_UE_sprite_height = 109
        self.eMBB_UE_screen_position_x = screen_position_x
        self.eMBB_UE_screen_position_y = screen_position_y
        self.original_x_pos = screen_position_x
        self.original_y_pos = screen_position_y
        self.filename = 'Resources/eMBB-UE-spritesheet.png'
        self.eMBB_UE_sprite = pygame.image.load(self.filename)
        #self.eMBB_UE_sprite.convert()
        self.sprite_surface = pygame.Surface((self.eMBB_UE_sprite_width,self.eMBB_UE_sprite_height))
        self.sprite_surface.set_colorkey((0,0,0))
        self.set_properties_eMBB()

        

        #self.sprite = SpriteSheet(self.spriteSheetFilename,self.spriteSheet_x,self.spriteSheet_y,self.spriteSheet_width,self.spriteSheet_height)
    def load_eMBB_UE_sprite(self,screen):
        self.sprite_surface.blit(self.eMBB_UE_sprite,(0,0))
        #screen.blit(self.sprite_surface,(self.eMBB_UE_screen_position_x,self.eMBB_UE_screen_position_y))

    def move_user(self,ENV_WIDTH,ENV_HEIGHT):
        self.eMBB_UE_screen_position_x = random.randint(self.xpos_move_lower_bound,self.xpos_move_upper_bound)
        self.eMBB_UE_screen_position_y = random.randint(self.ypos_move_lower_bound,self.ypos_move_upper_bound)

        if self.eMBB_UE_screen_position_x < 0 or self.eMBB_UE_screen_position_x > ENV_WIDTH:
            self.eMBB_UE_screen_position_x = self.original_x_pos

        if self.eMBB_UE_screen_position_y < 0 or self.eMBB_UE_screen_position_x > ENV_HEIGHT:
            self.eMBB_UE_screen_position_y = self.original_y_pos
        

    def generate_task(self,short_TTI,long_TTI):
        self.has_transmitted_this_time_slot = False
        self.timeslot_counter+=1

        #After every 1s generate packets with a uniform distribution between 5 and 10 packets per second
        #Long TTI = 0.125 ms. 1 second should be achieved after every 8000 timeslots
        if(self.timeslot_counter*long_TTI >= 1):
            self.timeslot_counter = 0

            #Require Task Arrival Rate, bits/packets, CPU cycles/packet
            self.task_arrival_rate_packets_per_second = random.randint(5,10) #Packets/s
            self.max_allowable_latency = random.randint(1000,2000) #[1,2] s
            self.max_allowable_reliability = 0
            self.packet_size = (random.randint(50,100))*8000 # [50,100]Kilobytes. 8000 bits in a KB
            self.QOS_requirement.set_requirements(self.max_allowable_latency,self.max_allowable_reliability)
            self.user_task.create_task(self.task_arrival_rate_packets_per_second,self.packet_size,self.QOS_requirement)
            self.communication_queue.append(self.user_task)
        

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos, Env_width_pixels, Env_width_metres):
        self.x_position = self.eMBB_UE_sprite.get_rect().centerx
        self.y_position = self.eMBB_UE_sprite.get_rect().centery

        x_diff_pixels = abs(SBS_x_pos-self.x_position)
        y_diff_pixels = abs(SBS_y_pos-self.y_position)

        x_diff_metres = (x_diff_pixels/Env_width_pixels)*Env_width_metres
        y_diff_metres = (y_diff_pixels/Env_width_pixels)*Env_width_metres

        self.distance_from_SBS = math.sqrt(math.pow(x_diff_metres,2)+math.pow(y_diff_metres,2))

    def collect_state(self):
        self.user_state_space.collect(self.total_gain,self.communication_queue,self.energy_harversted,self.QOS_requirement)
        return self.user_state_space

    def split_packet(self):
     
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                packet_dec = self.communication_queue[0].packet_queue[0]
                self.QOS_requirement_for_transmission = self.communication_queue[0].QOS_requirement
                packet_bin = bin(packet_dec)[2:]
                packet_size = len(packet_bin)
                self.packet_offload_size_bits = int(self.allocated_offloading_ratio*packet_size)
                self.packet_local_size_bits = int((1-self.allocated_offloading_ratio)*packet_size)
                self.local_queue.append(random.getrandbits(self.packet_local_size_bits))
                self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
                self.has_transmitted_this_time_slot = True
                self.dequeue_packet()
       

    def transmit_to_SBS(self, communication_channel, URLLC_Users):
        #Find URLLC users transmitting on this eMBB user's subcarriers
        subcarrier_URLLC_User_mapping = communication_channel.subcarrier_URLLC_User_mapping_
        last_subcarrier = subcarrier_URLLC_User_mapping[len(subcarrier_URLLC_User_mapping)-1][0]
        for subcarrier in self.allocated_subcarriers:
            self.intefering_URLLC_Users.append(subcarrier_URLLC_User_mapping[subcarrier - 1])
            if subcarrier == last_subcarrier:
                break

        #Calculate the bandwidth achieved on each subcarrier, each subcarrier receives interference from mapped URLLC users
        achieved_subcarriers_channel_rates = []
        first_subcarrier = self.allocated_subcarriers[0]
        for subcarrier in self.allocated_subcarriers:
            URLLC_users_on_this_subcarrier = self.intefering_URLLC_Users[subcarrier - first_subcarrier][1]
            URLLC_Users_transmit_powers = []
            URLLC_Users_channel_gains = []
            for URLLC_User in URLLC_users_on_this_subcarrier:
                if URLLC_Users[URLLC_User - 1].has_transmitted_this_time_slot == True:
                    URLLC_Users_transmit_powers.append(URLLC_Users[URLLC_User - 1].assigned_transmit_power_W)
                    URLLC_Users_channel_gains.append(URLLC_Users[URLLC_User - 1].total_gain)
            achieved_subcarrier_channel_rate = self.calculate_channel_rate(URLLC_Users_channel_gains,communication_channel)
            achieved_subcarriers_channel_rates.append(achieved_subcarrier_channel_rate)
            if subcarrier == last_subcarrier:
                break

        self.achieved_channel_rate = sum(achieved_subcarriers_channel_rates)

    def calculate_channel_rate(self,transmitting_URLLC_Users, communication_channel):
        channel_rate = communication_channel.subcarrier_bandwidth_kHz*(1-(len(transmitting_URLLC_Users)/communication_channel.num_minislots_per_timeslot))*math.log2(1+((self.assigned_transmit_power_W*self.total_gain)/(communication_channel.noise_spectral_density_W*communication_channel.subcarrier_bandwidth_kHz*1000)))
        '''print("embb user: ", self.eMBB_UE_label)
        print("communication_channel.subcarrier_bandwidth_kHz: ", communication_channel.subcarrier_bandwidth_kHz)
        print("len(transmitting_URLLC_Users): ", len(transmitting_URLLC_Users))
        print("communication_channel.num_minislots_per_timeslot: ", communication_channel.num_minislots_per_timeslot)
        print("(len(transmitting_URLLC_Users)/communication_channel.num_minislots_per_timeslot): ", (len(transmitting_URLLC_Users)/communication_channel.num_minislots_per_timeslot))
        print("self.assigned_transmit_power_W: ", self.assigned_transmit_power_W)
        print("self.total_gain: ", self.total_gain)
        print("self.assigned_transmit_power_W*self.total_gain: ", self.assigned_transmit_power_W*self.total_gain)
        print("communication_channel.noise_spectral_density_W: ", communication_channel.noise_spectral_density_W)
        print("communication_channel.subcarrier_bandwidth_kHz*1000: ",communication_channel.subcarrier_bandwidth_kHz*1000)
        print("math.log2(1+((self.assigned_transmit_power_W*self.total_gain)/(communication_channel.noise_spectral_density_W*communication_channel.subcarrier_bandwidth_kHz*1000)): ", math.log2(1+((self.assigned_transmit_power_W*self.total_gain)/(communication_channel.noise_spectral_density_W*communication_channel.subcarrier_bandwidth_kHz*1000))))
        print("channel rate: ", channel_rate)
        print("self.packet_offload_size_bits: ", self.packet_offload_size_bits )
        print(" ")
        print(" ")'''
        return channel_rate
    
    def local_processing(self):
        cycles_per_packet = self.cpu_cycles_per_byte*(self.packet_size*0.125)
        self.achieved_local_energy_consumption = self.energy_consumption_coefficient*math.pow(self.cpu_clock_frequency,2)*(1-self.allocated_offloading_ratio)*cycles_per_packet
        self.achieved_local_processing_delay = ((1-self.allocated_offloading_ratio)*cycles_per_packet)/self.cpu_clock_frequency
        self.local_queue.pop(0) 

    def offloading(self):
        self.achieved_transmission_delay = self.packet_offload_size_bits/self.achieved_channel_rate
        self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*self.achieved_transmission_delay

    def total_energy_consumed(self):
        self.achieved_total_energy_consumption = self.achieved_local_energy_consumption + self.achieved_transmission_energy_consumption

    def total_processing_delay(self):
        self.achieved_total_processing_delay = self.achieved_local_processing_delay + self.achieved_transmission_delay

    def set_matplotlib_rectangle_properties(self, communication_channel_long_TTI):
        for subcarrier in self.allocated_subcarriers:
            rectangle = Rectangle((0,subcarrier),communication_channel_long_TTI,1,color=(self.r,self.g,self.b,0.4))
            self.rectangles.append(rectangle)

    def random_color_generator(self):
        r = random.random()
        g = random.random()
        b = random.random()
        return (r,g,b)
    
    def set_properties_eMBB(self):
        self.max_allowable_latency = 2000 #[1,2] s
        self.max_allowable_reliability = 0
        self.QOS_requirement = QOS_requirement()
        self.QOS_requirement_for_transmission = QOS_requirement()
        self.packet_size_kilobytes = random.randint(50,100) #Kilobytes
        self.task_arrival_rate_packets_per_second = 0 #Packets/s
        self.user_task = Task(330)
        self.offloading_ratio = 0
        self.local_computation_delay_seconds = 0
        self.achieved_local_energy_consumption = 0
        self.offload_transmission_energy = 0
        self.battery_energy_level = 100 # Begin with 100%
        self.energy_harvested = 0
        self.achieved_transmission_delay = 0
        self.allocated_subcarriers = []
        self.number_of_allocated_subcarriers = 0
        self.local_queue = []
        self.timeslot_counter = 0
        self.minislot_counter = 0
        self.x_position = 0
        self.y_position = 0
        self.energy_harversted = 0
        self.user_state_space = State_Space(self.UE_label,self.total_gain,self.communication_queue,self.energy_harversted,self.QOS_requirement)
        self.allocated_offloading_ratio = 0
        self.packet_offload_size_bits = 0
        self.packet_local_size_bits = 0
        self.packet_size = 0
        self.intefering_URLLC_Users = []
        self.offloaded_packet = []
        self.single_side_standard_deviation_pos = 5
        self.xpos_move_lower_bound = self.eMBB_UE_screen_position_x - self.single_side_standard_deviation_pos
        self.xpos_move_upper_bound = self.eMBB_UE_screen_position_x + self.single_side_standard_deviation_pos
        self.ypos_move_lower_bound = self.eMBB_UE_screen_position_y - self.single_side_standard_deviation_pos
        self.ypos_move_upper_bound = self.eMBB_UE_screen_position_y + self.single_side_standard_deviation_pos
        self.rectangles = []
        self.r,self.g,self.b = self.random_color_generator()
        self.min_communication_qeueu_size = 0
        self.max_communication_qeueu_size = 8000000






  



            


