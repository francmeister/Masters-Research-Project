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
            for task in self.communication_queue:
                com_queue_size += len(task.packet_queue)

        return com_queue_size
