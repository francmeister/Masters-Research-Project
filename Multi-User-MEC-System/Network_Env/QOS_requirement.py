import pygame, sys, time, random
import random
pygame.init()

class QOS_requirement():
    def __init__(self,max_allowable_latency, max_allowable_reliability):
        self.max_allowable_latency = max_allowable_latency
        self.max_allowable_reliability = max_allowable_reliability

    def set_requirements(self,max_allowable_latency, max_allowable_reliability):
        self.max_allowable_latency = max_allowable_latency
        self.max_allowable_reliability = max_allowable_reliability