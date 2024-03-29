import pygame, sys, time, random
from Spritesheet import SpriteSheet
pygame.init()
from numpy import interp
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
        self.x_position = 0
        self.y_position = 0

        self.cell_tower_sprite1 = pygame.image.load(self.filename1)
        #self.cell_tower_sprite1.convert()
        self.cell_tower_sprite2 = pygame.image.load(self.filename2)
        #self.cell_tower_sprite2.convert()
        self.cell_tower_sprite3 = pygame.image.load(self.filename3)
        #self.cell_tower_sprite3.convert()
        self.cell_tower_sprite4 = pygame.image.load(self.filename4)
        #self.cell_tower_sprite4.convert()

        self.sprite_surface = pygame.Surface((self.cell_tower_sprite_width,self.cell_tower_sprite_height))
        self.individual_rewards = []
        self.sprite_surface.set_colorkey((0,0,0))
        self.set_properties()
                
    def load_cell_tower_sprite(self,screen,SCREEN_WIDTH,SCREEN_HEIGHT,frameCount):
        if frameCount == 0:
            self.sprite_surface.blit(self.cell_tower_sprite1,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            #screen.blit(self.sprite_surface,(400,0))

        elif frameCount == 1:
            self.sprite_surface.blit(self.cell_tower_sprite2,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            #screen.blit(self.sprite_surface,(400,0))


        elif frameCount == 2:
            self.sprite_surface.blit(self.cell_tower_sprite3,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            #screen.blit(self.sprite_surface,(400,0))

        elif frameCount == 3:
            self.sprite_surface.blit(self.cell_tower_sprite4,(0,0))
            #screen.blit(sprite_surface,(SCREEN_WIDTH/2-self.cell_tower_sprite_width/2,SCREEN_HEIGHT/2-self.cell_tower_sprite_height))
            #screen.blit(self.sprite_surface,(400,0))

    def associate_users(self, eMBB_Users, URLLC_Users):
        self.associated_eMBB_users = eMBB_Users
        self.associated_URLLC_users = URLLC_Users

    def get_SBS_center_pos(self):
        self.x_position = self.cell_tower_sprite1.get_rect().centerx
        self.y_position = self.cell_tower_sprite1.get_rect().centery

    def collect_state_space(self, eMBB_Users, URLLC_Users):
        Users = eMBB_Users + URLLC_Users
        self.system_state_space.clear()
        channel_gains = []
        communication_queue_size = []
        energy_harvested = []
        latency_requirement = []
        reliability_requirement = []
        #Collect Channel gains
        for user in Users:
            channel_gains.append(user.user_state_space.channel_gain)
            communication_queue_size.append(user.user_state_space.calculate_communication_queue_size())
            energy_harvested.append(user.user_state_space.energy_harvested)
            latency_requirement.append(user.user_state_space.QOS_requirements.max_allowable_latency)
            reliability_requirement.append(user.user_state_space.QOS_requirements.max_allowable_reliability)

        self.system_state_space.append(channel_gains)
        self.system_state_space.append(communication_queue_size)
        self.system_state_space.append(energy_harvested)
        self.system_state_space.append(latency_requirement)
        self.system_state_space.append(reliability_requirement)
        return self.system_state_space

    def allocate_transmit_powers(self,eMBB_Users, action):
        index = 0
        for User in eMBB_Users:
            User.assigned_transmit_power_dBm = action[index]
            User.calculate_assigned_transmit_power_W()
            index+=1

    def allocate_offlaoding_ratios(self,eMBB_Users, action):
        index = 0
        for eMBB_User in eMBB_Users:
            eMBB_User.allocated_offloading_ratio = action[index]
            index+=1

    def count_num_arriving_URLLC_packet(self,URLLC_Users):
        self.num_arriving_URLLC_packets = 0

        for URLLC_User in URLLC_Users:
            if URLLC_User.has_transmitted_this_time_slot == True:
                self.num_arriving_URLLC_packets += 1

    def receive_offload_packets(self, eMBB_Users, URLLC_Users):
        for eMBB_User in eMBB_Users:
            if eMBB_User.has_transmitted_this_time_slot == True:
                self.eMBB_Users_packet_queue.append(eMBB_User.offloaded_packet)

        for URLLC_User in URLLC_Users:
            if URLLC_User.has_transmitted_this_time_slot == True:
                self.eMBB_Users_packet_queue.append(eMBB_User.offloaded_packet)

    def calculate_achieved_total_system_energy_consumption(self, eMBB_Users):
        self.total_system_energy_consumption = 0
        for eMBB_User in eMBB_Users:
            self.achieved_total_system_energy_consumption += eMBB_User.achieved_total_energy_consumption

        #print("achieved_total_system_energy_consumption", self.achieved_total_system_energy_consumption)

    def calculate_achieved_total_system_processing_delay(self, eMBB_Users):
        self.achieved_total_system_processing_delay = 0
        for eMBB_User in eMBB_Users:
            self.achieved_total_system_processing_delay += eMBB_User.achieved_total_processing_delay

    def calculate_achieved_total_rate_URLLC_users(self, URLLC_Users):
        self.achieved_total_rate_URLLC_users = 0
        for URLLC_User in URLLC_Users:
            if URLLC_User.has_transmitted_this_time_slot == True:
                self.achieved_total_rate_URLLC_users += URLLC_User.achieved_channel_rate

    def calculate_achieved_total_rate_eMBB_users(self, eMBB_Users):
        self.achieved_total_rate_eMBB_users = 0
        for eMBB_User in eMBB_Users:
            if eMBB_User.has_transmitted_this_time_slot == True:
                self.achieved_total_rate_eMBB_users += eMBB_User.achieved_channel_rate

    def calculate_achieved_URLLC_reliability(self, URLLC_Users):
        self.achieved_URLLC_reliability = URLLC_Users[0].packet_size_bits*self.num_arriving_URLLC_packets

    def calculate_achieved_system_energy_efficiency(self):
        if self.achieved_total_system_energy_consumption == 0:
            self.achieved_system_energy_efficiency = 0
        else:
            self.achieved_system_energy_efficiency = self.achieved_total_rate_eMBB_users/self.achieved_total_system_energy_consumption

        #print("self.achieved_system_energy_efficiency",self.achieved_system_energy_efficiency)

    def calculate_achieved_system_reward(self, eMBB_Users, URLLC_Users):
        self.achieved_system_reward = 0
        eMBB_User_energy_consumption = 0
        eMBB_User_channel_rate = 0
        eMBB_User_QOS_requirement_revenue_or_penelaty = 0
        total_energy = 0
        total_rate = 0
        total_QOS_revenue = 0
        self.individual_rewards.clear()
        for eMBB_User in eMBB_Users:
            eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption 
            total_energy += eMBB_User_energy_consumption
            eMBB_User_channel_rate = eMBB_User.achieved_channel_rate
            #print('eMBB_User_channel_rate')
            #print(eMBB_User_channel_rate)
            eMBB_User_channel_rate = interp(eMBB_User_channel_rate,[60000000,153000000],[0,100])
            total_rate += eMBB_User_channel_rate
            #eMBB_User_QOS_requirement_revenue_or_penelaty = self.achieved_eMBB_delay_requirement_revenue_or_penalty(eMBB_User)
            #total_QOS_revenue += eMBB_User_QOS_requirement_revenue_or_penelaty
            if eMBB_User_energy_consumption == 0:
                individual_reward = 0
            else:
                individual_reward = eMBB_User_energy_consumption#eMBB_User_channel_rate#eMBB_User_energy_consumption #+ eMBB_User_channel_rate + eMBB_User_QOS_requirement_revenue_or_penelaty
            self.achieved_system_reward += individual_reward
            self.individual_rewards.append(individual_reward)

        #print("total_energy: ", total_energy)
        #print("total_rate: ", total_rate)
        #print("total_QOS_revenue: ", total_QOS_revenue)

        if self.num_arriving_URLLC_packets > 0:
           self.achieved_system_reward += ((self.achieved_total_rate_URLLC_users-URLLC_Users[0].QOS_requirement_for_transmission.max_allowable_reliability)/self.num_arriving_URLLC_packets)

        for URLLC_user in URLLC_Users:
           self.individual_rewards.append(0)
        
        return self.achieved_system_reward, self.individual_rewards, total_energy,total_rate

    def achieved_eMBB_delay_requirement_revenue_or_penalty(self,eMBB_User):
        processing_delay_requirement = eMBB_User.QOS_requirement_for_transmission.max_allowable_latency
        achieved_local_processing_delay = eMBB_User.achieved_local_processing_delay
        achieved_offload_processing_delay = eMBB_User.achieved_transmission_delay

        if processing_delay_requirement - max(achieved_local_processing_delay,achieved_offload_processing_delay) >= 0:
            return self.eMBB_User_delay_requirement_revenue
        else:
            return (processing_delay_requirement - max(achieved_local_processing_delay,achieved_offload_processing_delay))
        
    def achieved_URLLC_User_reliability_requirement_revenue_or_penalty(self,URLLC_User):
        reliability_requirement = URLLC_User.QOS_requirement_for_transmission.max_allowable_reliability
        achieved_reliability = self.achieved_total_rate_URLLC_users

        if ((achieved_reliability-reliability_requirement)/self.num_arriving_URLLC_packets) >= 0:
            return self.eMBB_User_delay_requirement_revenue
        else:
            return ((achieved_reliability-reliability_requirement)/self.num_arriving_URLLC_packets)
        
    #def perform_timeslot_sequential_events(self,eMBB_Users,URLLC_Users,communication_channel):

    def set_properties(self):
        self.associated_users = []
        self.associated_URLLC_users = []
        self.associated_eMBB_users = []
        self.system_state_space = []
        self.num_arriving_URLLC_packets = 0
        self.eMBB_Users_packet_queue = []
        self.URLLC_Users_packet_queue = []
        self.achieved_total_system_energy_consumption = 0
        self.achieved_total_system_processing_delay = 0
        self.achieved_URLLC_reliability = 0
        self.achieved_total_rate_URLLC_users = 0
        self.achieved_total_rate_eMBB_users = 0
        self.achieved_system_energy_efficiency = 0
        self.achieved_system_reward = 0
        self.eMBB_User_delay_requirement_revenue = 5
        self.URLLC_User_reliability_requirement_revenue = 5
        self.clock_frequency = 2
        self.work_load = 0
        self.eMBB_UEs = []
        self.URLLC_UEs = []
        self.achieved_users_energy_consumption = 0
        self.achieved_users_channel_rate = 0






