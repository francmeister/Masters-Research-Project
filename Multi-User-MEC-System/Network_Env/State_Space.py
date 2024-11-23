from Task import Task
from QOS_requirement import QOS_requirement

class State_Space():
    def __init__(self):
        self.User_label = 0
        self.channel_gain = 0
        self.communication_queue = []
        self.communication_queue_size = self.calculate_communication_queue_size()
        self.battery_energy = 0
        self.QOS_requirements = QOS_requirement(10,0)
        self.local_cpu_frequency = 0
        self.offloading_queue_length = 0
        self.local_queue_length = 0

    def collect(self,channel_gain,battery_energy,offLoad_queue_length,local_queue_length):
        self.channel_gain = channel_gain
        #self.communication_queue = communication_queue
        self.battery_energy = battery_energy
        self.offloading_queue_length = offLoad_queue_length
        self.local_queue_length = local_queue_length
        #print('battery energy: ', self.battery_energy)
        #self.QOS_requirements = QOS_requirements
        #self.local_cpu_frequency = local_cpu_frequency

    def collect_urrlc(self,channel_gain):
        self.channel_gain = channel_gain


    def calculate_communication_queue_size(self):
        com_queue_size = len(self.communication_queue)

        return com_queue_size
    
