
import random
import numpy as np
import math


class User_Equipment():
    def __init__(self):
        self.set_properties_UE()

    def calculate_channel_gain(self):
        #Pathloss gain
        self.pathloss_gain = (math.pow(10,(35.3+37.6*math.log10(self.distance_from_SBS))))/10
        self.small_scale_channel_gain = np.random.exponential(1)
        #self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        self.total_gain = self.small_scale_channel_gain #*self.large_scale_channel_gain*self.pathloss_gain

    def calculate_assigned_transmit_power_W(self):
        self.assigned_transmit_power_W = self.assigned_transmit_power_dBm#(math.pow(10,(self.assigned_transmit_power_dBm/10)))/1000

    def dequeue_packet(self):
        if len(self.communication_queue) > 0:
            if len(self.communication_queue[0].packet_queue) > 0:
                self.communication_queue[0].packet_queue.pop(0)

            elif len(self.communication_queue[0].packet_queue) == 0:
                self.dequeue_task()

    def dequeue_task(self):
        self.communication_queue.pop(0)

    def set_properties_UE(self):
        self.max_transmission_power_dBm = 23 # dBm
        self.max_transmission_power_W =  (math.pow(10,(self.max_transmission_power_dBm/10)))/1000# Watts
        self.assigned_transmit_power_dBm = 0
        self.assigned_transmit_power_W = 0
        self.cpu_cycles_per_byte = 330
        self.cpu_clock_frequency = 5000 #cycles/slot
        self.associated_SBS_label = 0
        self.small_scale_channel_gain = 0
        self.large_scale_channel_gain = 0
        self.pathloss_gain = 0
        self.achieved_channel_rate = 0
        self.achieved_SNIR = 0
        self.task_profile = []
        self.distance_from_SBS = 0
        #self.total_gain = np.zeros(self.communication_channel.num_allocate_RBs_upper_bound)
        self.has_transmitted_this_time_slot = False
        self.communication_queue = []
        self.energy_consumption_coefficient = math.pow(10,-15)
        self.achieved_transmission_energy_consumption = 0
        self.achieved_local_processing_delay = 0
        self.achieved_total_energy_consumption = 0
        self.achieved_total_processing_delay = 0
        self.min_channel_gain = -100
        self.max_channel_gain = 100
        self.max_energy_harvested = 100
    