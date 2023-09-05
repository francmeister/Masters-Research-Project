import pygame, sys, time, random
import random
from QOS_requirement import QOS_requirement
pygame.init()
import numpy as np

class Task():
    def __init__(self,CPU_cycles_per_byte):
        self.task_arrival_rate = 0
        self.bits_per_packet = 0
        self.number_of_bits_in_byte = 8
        self.CPU_cycles_per_byte = CPU_cycles_per_byte
        self.cycles_per_packet = 0
        self.QOS_requirement = QOS_requirement()
        self.packet_queue = []

    def create_task(self,task_arrival_rate,bits_per_packet, QOS_requirement):
        self.task_arrival_rate = task_arrival_rate
        self.bits_per_packet = bits_per_packet
        self.cycles_per_packet = (self.CPU_cycles_per_byte/self.number_of_bits_in_byte)*self.bits_per_packet
        self.QOS_requirement = QOS_requirement
        self.packet_queue = []

        for i in range(self.task_arrival_rate - 1):
            #print('self.bits_per_packet: ', self.bits_per_packet)
            self.packet_queue.append(random.getrandbits(self.bits_per_packet))