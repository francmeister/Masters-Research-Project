from Task import Task

class State_Space():
    def __init__(self, User_label,channel_gain,communication_queue,energy_harvested, QOS_requirements):
        self.User_label = User_label
        self.channel_gain = channel_gain
        self.communication_queue = communication_queue
        self.communication_queue_size = self.calculate_communication_queue_size()
        self.energy_harvested = energy_harvested
        self.QOS_requirements = QOS_requirements

    def collect(self,channel_gain,communication_queue,energy_harvested, QOS_requirements):
        self.channel_gain = channel_gain
        self.communication_queue = communication_queue
        self.energy_harvested = energy_harvested
        self.QOS_requirements = QOS_requirements

    def calculate_communication_queue_size(self):
        com_queue_size = 0
        if len(self.communication_queue) > 0:
            packet_dec = self.communication_queue[0].packet_queue[0]
            packet_bin = bin(packet_dec)[2:]
            com_queue_size = len(packet_bin)

        return com_queue_size
