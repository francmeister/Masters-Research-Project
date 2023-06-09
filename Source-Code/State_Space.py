from Task import Task

class State_Space():
    def __init__(self, User_label,channel_gain,communication_queue,energy_harvested):
        self.User_label = User_label
        self.channel_gain = channel_gain
        self.communication_queue = communication_queue
        self.energy_harvested = energy_harvested

    def collect(self,channel_gain,communication_queue,energy_harvested):
        self.channel_gain = channel_gain
        self.communication_queue = communication_queue
        self.energy_harvested = energy_harvested
