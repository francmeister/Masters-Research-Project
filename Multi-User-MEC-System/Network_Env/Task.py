import pygame, sys, time, random
import random
from QOS_requirement import QOS_requirement
pygame.init()
import numpy as np

class Task():
    def __init__(self,CPU_cycles_per_byte,slot_task_size_bits, QOS_requirement, queue_timer, task_identifier):
        self.slot_task_size = slot_task_size_bits
        self.CPU_cycles_per_byte = CPU_cycles_per_byte
        self.cycles_per_bit = self.CPU_cycles_per_byte/8
        self.required_computation_cycles = self.cycles_per_bit*self.slot_task_size
        self.QOS_requirement = QOS_requirement
        self.bits = random.getrandbits(self.slot_task_size)
        self.queue_timer = queue_timer
        self.task_identifier = task_identifier
        self.local_queue_timer = 0
        self.offload_queue_timer = 0
        

    #Specify slot task size, computation cycles and latency requirement
    def create_task(self,slot_task_size_bits, cycles, QOS_requirement, queue_timer, task_identifier):
        self.slot_task_size = slot_task_size_bits
        self.required_computation_cycles = cycles
        self.QOS_requirement = QOS_requirement
        self.bits = random.getrandbits(self.slot_task_size)
        self.queue_timer = queue_timer
        self.task_identifier = task_identifier

    def increment_queue_timer(self):
        self.queue_timer+=1

    def split_task(self, bits_amount_processed):
        if self.slot_task_size > bits_amount_processed:
            remaining_bits = self.slot_task_size - bits_amount_processed
            self.slot_task_size = int(remaining_bits)
            self.required_computation_cycles = (self.CPU_cycles_per_byte/8)*self.slot_task_size
            self.bits = random.getrandbits(self.slot_task_size)
        #else:
            #print('self.slot_task_size: ', self.slot_task_size, 'bits_amount_processed: ', bits_amount_processed)
            #print('self.required_computation_cycles: ', self.required_computation_cycles, 'remaining cycles: ', bits_amount_processed*(self.CPU_cycles_per_byte/8))
            #print('error------------------------------------------------------------------------------------------------------------------------------------------------------------')

