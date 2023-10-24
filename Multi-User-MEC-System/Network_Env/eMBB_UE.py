import pygame, sys, time, random
import random
import numpy as np
from User_Equipment import User_Equipment
from QOS_requirement import QOS_requirement
from Task import Task
import numpy as np
from matplotlib.patches import Rectangle
import math
from State_Space import State_Space
from numpy import interp
import pandas as pd
from Communication_Channel import Communication_Channel

class eMBB_UE(User_Equipment):
    def __init__(self, eMBB_UE_label,x,y):
        #User_Equipment.__init__(self)
        self.UE_label = eMBB_UE_label
        self.original_x_position = x
        self.original_y_position = y
        self.eMBB_UE_label = eMBB_UE_label
        self.communication_channel = Communication_Channel(1)
        self.set_properties_eMBB()

    def set_properties_eMBB(self):

        #State Space Limits
        self.max_allowable_latency = 2000 #[1,2] s
        self.min_allowable_latency = 1000

        self.max_allowable_reliability = 0

        self.min_communication_qeueu_size = 0
        self.max_communication_qeueu_size = 50

        self.min_channel_gain = math.pow(10,-5)
        self.max_channel_gain = 10

        self.min_energy_harvested = 0
        self.max_energy_harvested = 150

        self.max_battery_energy = 22000
        self.min_battery_energy = 0

        self.max_cpu_frequency = 5000
        self.min_cpu_frequency = 5

        self.max_task_size_KB_per_second = 100 #100KB per second
        self.min_task_size_KB_per_second = 50 #50KB per second

        self.max_queue_length_KBs = math.pow(self.max_task_size_KB_per_second,3) # 1GB
        self.min_queue_length_KBs = 0

        self.max_task_arrival_rate_tasks_per_second = 10
        self.min_task_arrival_rate_tasks_per_second = 5

        self.max_queue_length_number = self.calculate_max_queue_length_number(self.communication_channel,self.max_task_arrival_rate_tasks_per_second)
        self.min_queue_length = 0

        self.battery_energy_level = (random.randint(15000,25000))

        self.cycles_per_byte = 330
        self.cycles_per_bit = self.cycles_per_byte/8
        self.max_service_rate_cycles_per_slot = 620000#5000

        #self.QOS_requirement = QOS_requirement()
        #self.QOS_requirement_for_transmission = QOS_requirement()
        #self.user_task = Task(330)
        #self.local_task = Task(330)
        #self.offload_task = Task(330)
        self.local_computation_delay_seconds = 0
        self.achieved_local_energy_consumption = 0
        self.offload_transmission_energy = 0
        #self.battery_energy_level = 100 # Begin with 100%
        self.energy_harvested = 0
        self.achieved_transmission_delay = 0
        self.local_queue = []
        self.timeslot_counter = 0
        self.x_position = self.original_x_position
        self.y_position = self.original_y_position
        self.energy_harversted = 0
        self.distance_from_SBS = 0
        self.total_gain = 0
        self.has_transmitted_this_time_slot = False
        self.communication_queue = []
        self.energy_consumption_coefficient = math.pow(10,-15)
        self.achieved_transmission_energy_consumption = 0
        self.achieved_local_processing_delay = 0
        self.achieved_total_energy_consumption = 0
        self.achieved_total_processing_delay = 0
        self.cpu_cycles_per_byte = 330
        self.cpu_clock_frequency = (random.randint(5,5000)) #cycles/slot
        self.user_state_space = State_Space()
        self.allocated_offloading_ratio = 0
        self.packet_offload_size_bits = 0
        self.packet_local_size_bits = 0
        self.packet_size = 0
        self.delay_reward = 10
        self.achieved_channel_rate_normalized = 0
        self.achieved_total_energy_consumption_normalized = 0
        self.dequeued_local_tasks = []
        self.dequeued_offload_tasks = []
        self.completed_tasks = []
   
    
        self.single_side_standard_deviation_pos = 5
        self.xpos_move_lower_bound = self.x_position - self.single_side_standard_deviation_pos
        self.xpos_move_upper_bound = self.x_position + self.single_side_standard_deviation_pos
        self.ypos_move_lower_bound = self.y_position - self.single_side_standard_deviation_pos
        self.ypos_move_upper_bound = self.y_position + self.single_side_standard_deviation_pos
        self.allocated_RBs = []

        self.max_transmission_power_dBm = 70 # dBm
        self.min_transmission_power_dBm = 0
        self.max_transmission_power_W =  (math.pow(10,(self.max_transmission_power_dBm/10)))/1000# Watts
        self.min_transmission_power_W =  (math.pow(10,(self.min_transmission_power_dBm/10)))/1000# Watts
        self.assigned_transmit_power_dBm = 0
        self.assigned_transmit_power_W = 0
        
       
        self.small_scale_channel_gain = 0
        self.large_scale_channel_gain = 0
        self.pathloss_gain = 0
        self.achieved_channel_rate = 0
        self.allowable_latency = 0
        self.task_queue = []
        self.task_identifier = 0
        self.task_arrival_rate_tasks_per_second = 0
        self.ptr = 0
        self.queuing_delay = 0
        self.previous_slot_battery_energy = 0

    def move_user(self,ENV_WIDTH,ENV_HEIGHT):
        self.x_position = random.randint(self.xpos_move_lower_bound,self.xpos_move_upper_bound)
        self.y_position = random.randint(self.ypos_move_lower_bound,self.ypos_move_upper_bound)

        if self.x_position < 0 or self.x_position > ENV_WIDTH:
            self.x_position = self.original_x_position

        if self.y_position < 0 or self.y_position > ENV_HEIGHT:
            self.y_position = self.original_y_position
        

    def calculate_max_queue_length_number(self,communication_channel,max_task_arrival_rate_tasks_per_second):
        max_task_size_per_second_kilobytes = self.max_task_size_KB_per_second # 100 kilobytes
        max_task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*max_task_arrival_rate_tasks_per_second
        max_task_size_per_slot_kilobytes = int(max_task_size_per_second_kilobytes*max_task_arrival_rate_tasks_slot)

        return (self.max_queue_length_KBs/max_task_size_per_slot_kilobytes)

    def generate_task(self,communication_channel):
        self.has_transmitted_this_time_slot = False
        self.timeslot_counter+=1

        #Specify slot task size, computation cycles and latency requirement
        #self.task_arrival_rate_tasks_per_second = random.randint(self.min_task_arrival_rate_tasks_per_second,self.max_task_arrival_rate_tasks_per_second)
        self.task_arrival_rate_tasks_per_second = np.random.poisson(5,1)
        self.task_arrival_rate_tasks_per_second = self.task_arrival_rate_tasks_per_second[0]
        qeueu_timer = 0


        if len(self.task_queue) >= self.max_queue_length_number:
            for x in range(0,self.task_arrival_rate_tasks_per_second):
                #np.random.poisson(10)
                #task_size_per_second_kilobytes = random.randint(self.min_task_size_KB_per_second,self.max_task_size_KB_per_second) #choose between 50 and 100 kilobytes
                #task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*self.task_arrival_rate_tasks_per_second
                #task_size_per_slot_kilobytes = task_size_per_second_kilobytes*task_arrival_rate_tasks_slot
                task_size_per_slot_bits = int(np.random.uniform(500,1500))#Average of 1000 bits per task in slot #int(task_size_per_slot_kilobytes*8000) #8000 bits in a KB----------
                #task_cycles_required = self.cycles_per_bit*task_size_per_slot_bits#-------------
                latency_requirement = 10#latency required is 10 ms for every task#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
                reliability_requirement = 0
                QOS_requirement_ = QOS_requirement(latency_requirement,reliability_requirement)
                user_task = Task(330,task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
                self.task_identifier+=1
                #print('task identifier: ', self.task_identifier)

                self.storage[int(self.ptr)] = user_task
                self.ptr = (self.ptr + 1) % self.max_queue_length_number
        else:
            for x in range(0,self.task_arrival_rate_tasks_per_second):
                #task_size_per_second_kilobytes = random.randint(self.min_task_size_KB_per_second,self.max_task_size_KB_per_second) #choose between 50 and 100 kilobytes
                #task_arrival_rate_tasks_slot = (communication_channel.long_TTI/1000)*self.task_arrival_rate_tasks_per_second
                #task_size_per_slot_kilobytes = task_size_per_second_kilobytes*task_arrival_rate_tasks_slot
                task_size_per_slot_bits = int(np.random.uniform(500,1500)) #8000 bits in a KB----------
                #task_cycles_required = self.cycles_per_bit*task_size_per_slot_bits#-------------
                latency_requirement = 10#random.randint(self.min_allowable_latency,self.max_allowable_latency) #[1,2] s
                reliability_requirement = 0
                QOS_requirement_ = QOS_requirement(latency_requirement,reliability_requirement)
                user_task = Task(330,task_size_per_slot_bits,QOS_requirement_,qeueu_timer,self.task_identifier)
                self.task_identifier+=1
                #print('task identifier: ', self.task_identifier)
                self.task_queue.append(user_task)
        

    def calculate_distance_from_SBS(self, SBS_x_pos, SBS_y_pos, Env_width_pixels, Env_width_metres):

        x_diff_metres = abs(SBS_x_pos-self.x_position)
        y_diff_metres = abs(SBS_y_pos-self.y_position)


        self.distance_from_SBS = math.sqrt(math.pow(x_diff_metres,2)+math.pow(y_diff_metres,2))

    def collect_state(self):
            #self.cpu_clock_frequency = (random.randint(5,5000))
            self.user_state_space.collect(self.total_gain,self.previous_slot_battery_energy)
            #self.user_state_space.collect(self.total_gain,self.communication_queue,self.battery_energy_level,self.communication_queue[0].QOS_requirement,self.cpu_clock_frequency)
            return self.user_state_space

    def split_tasks(self):
        if len(self.task_queue) > 0:
            task_identities = []
            task_sizes_bits = []
            required_cycles = []
            latency_requirements = []

            if len(self.task_queue) > 0:
                for task in self.task_queue:
                    task_identities.append(task.task_identifier)
                    latency_requirements.append(task.QOS_requirement.max_allowable_latency)
                    task_sizes_bits.append(task.slot_task_size)
                    required_cycles.append(task.required_computation_cycles)

            data = {
                'Task Identity':task_identities,
                'Task Size Bits':task_sizes_bits,
                #'Required Cycles':required_cycles,
                'Latency requirement':latency_requirements
            }

            df = pd.DataFrame(data=data)
            #print('--------------------------------------------Timeslot: ',self.timeslot_counter, '--------------------------------------------')
            #print('task queue data')
            #print(df)
            #print(' ')

            for x in range(0,self.task_arrival_rate_tasks_per_second):
                packet_dec = self.task_queue[x].bits
                self.QOS_requirement_for_transmission = self.task_queue[x].QOS_requirement
                packet_bin = bin(packet_dec)[2:]
                packet_size = len(packet_bin)
                self.packet_offload_size_bits = int(self.allocated_offloading_ratio*packet_size)
                self.packet_local_size_bits = int((1-self.allocated_offloading_ratio)*packet_size)
    
                local_task = Task(330,self.packet_local_size_bits,self.task_queue[x].QOS_requirement,self.task_queue[x].queue_timer,self.task_queue[x].task_identifier)
                offload_task = Task(330,self.packet_offload_size_bits,self.task_queue[x].QOS_requirement,self.task_queue[x].queue_timer,self.task_queue[x].task_identifier)

                self.local_queue.append(local_task)
                self.communication_queue.append(offload_task)

                #self.offloaded_packet = random.getrandbits(self.packet_offload_size_bits)
                #self.has_transmitted_this_time_slot = True
            for x in range(0,self.task_arrival_rate_tasks_per_second):
                self.task_queue.pop(0)

            if(self.allocated_offloading_ratio > 0):
                self.has_transmitted_this_time_slot = True
            #print('local qeueu length: ', len(self.local_queue))
            #print('offload qeueu length: ', len(self.communication_queue))
            #print(' ')

            local_task_identities = []
            local_task_sizes_bits = []
            local_required_cycles = []
            local_latency_requirements = []

            if len(self.local_queue) > 0:
                for task in self.local_queue:
                    local_task_identities.append(task.task_identifier)
                    local_task_sizes_bits.append(task.slot_task_size)
                    local_required_cycles.append(task.required_computation_cycles)
                    local_latency_requirements.append(task.QOS_requirement.max_allowable_latency)

            local_data = {
                'Task Identity':local_task_identities,
                'Task Size Bits':local_task_sizes_bits,
                #'Required Cycles':local_required_cycles,
                'Latency requirement':local_latency_requirements
            }

            df = pd.DataFrame(data=local_data)
            #print('local queue data')
            #print(df)
            #print(' ')

            offload_task_identities = []
            offload_task_sizes_bits = []
            offload_required_cycles = []
            offload_latency_requirements = []

            if len(self.communication_queue) > 0:
                for task in self.communication_queue:
                    offload_task_identities.append(task.task_identifier)
                    offload_task_sizes_bits.append(task.slot_task_size)
                    offload_required_cycles.append(task.required_computation_cycles)
                    offload_latency_requirements.append(task.QOS_requirement.max_allowable_latency)

            offload_data = {
                'Task Identity':offload_task_identities,
                'Task Size Bits':offload_task_sizes_bits,
                #'Required Cycles':offload_required_cycles,
                'Latency requirement':offload_latency_requirements
            }

            df = pd.DataFrame(data=offload_data)
            #print('offload queue data')
            #print(df)
            #print(' ')
            #print('Size of offloading queue: ',sum(offload_task_sizes_bits))
       

    def transmit_to_SBS(self, communication_channel):
        #Calculate the bandwidth achieved on each RB
        achieved_RB_channel_rates = []
        #print('number of allocated RBs: ', len(self.allocate(d_RBs))
        if (len(self.allocated_RBs)  > 0) and (self.battery_energy_level > 0):
            for RB in self.allocated_RBs:
                achieved_RB_channel_rate = self.calculate_channel_rate(communication_channel)
                achieved_RB_channel_rates.append(achieved_RB_channel_rate)
            #print('channel gain: ', self.total_gain, " achieved channel rate matrix: ", achieved_RB_channel_rates)
            self.achieved_channel_rate = sum(achieved_RB_channel_rates)
            min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
            self.achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[0,15000],[0,1])
        else:
            self.achieved_channel_rate = 0
            self.achieved_channel_rate_normalized = 0

        #print('achieved channel rate: ', self.achieved_channel_rate)
        #print(' ')

    def calculate_channel_rate(self, communication_channel):
        RB_bandwidth = communication_channel.RB_bandwidth_Hz
        noise_spectral_density = communication_channel.noise_spectral_density_W
        channel_rate_numerator = self.assigned_transmit_power_W*self.total_gain
        channel_rate_denominator = noise_spectral_density#*RB_bandwidth
        channel_rate = RB_bandwidth*math.log2(1+(channel_rate_numerator/channel_rate_denominator))
        return (channel_rate/1000)
    
    def local_processing(self):
        cpu_cycles_left = self.max_service_rate_cycles_per_slot #check if 
        self.achieved_local_energy_consumption = 0
        self.dequeued_local_tasks.clear()
        counter = 0
        
        for local_task in self.local_queue:
            #print('cycles left: ', cpu_cycles_left)
            #print('local_task.required_computation_cycles: ', local_task.required_computation_cycles)
            if cpu_cycles_left > local_task.required_computation_cycles:
                #print('cycles left: ', cpu_cycles_left)
                self.achieved_local_energy_consumption += self.energy_consumption_coefficient*math.pow(local_task.required_computation_cycles,2)*local_task.required_computation_cycles
                cpu_cycles_left-=local_task.required_computation_cycles
                self.dequeued_local_tasks.append(local_task)
                counter += 1

            elif cpu_cycles_left < local_task.required_computation_cycles and cpu_cycles_left > self.cycles_per_bit:
                bits_that_can_be_processed = cpu_cycles_left/self.cycles_per_bit
                self.achieved_local_energy_consumption += self.energy_consumption_coefficient*math.pow(cpu_cycles_left,2)*cpu_cycles_left
                local_task.split_task(bits_that_can_be_processed) 
                break

        for x in range(0,counter):
            self.local_queue.pop(0)

        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        dequeued_task_size = []
        total_sum_size_dequeued_tasks = []

        if len(self.dequeued_local_tasks) > 0:
            for dequeued_local_task in self.dequeued_local_tasks:
                task_identities.append(dequeued_local_task.task_identifier)
                task_latency_requirements.append(dequeued_local_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(dequeued_local_task.queue_timer)
                dequeued_task_size.append(dequeued_local_task.slot_task_size)

            for dequeued_local_task in self.dequeued_local_tasks:
                total_sum_size_dequeued_tasks.append(sum(dequeued_task_size))
                  

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency,
                "Size of Dequeued Task":dequeued_task_size,
                "Sum size of all Dequeued Tasks":total_sum_size_dequeued_tasks
            }

            df = pd.DataFrame(data=data)

            #print('Dequeued Local Tasks')
            #print(df)
            #print(' ')
            #print('Local Computation energy consumed in this slot: ', self.achieved_local_energy_consumption)
            #print(' ')

        #print(' ')
        #print(' ')

        min_local_energy_consumption, max_local_energy_consumption = self.min_and_max_achievable_local_energy_consumption()
        min_local_computation_delay, max_local_computation_delay = self.min_max_achievable_local_processing_delay()
        #print('min local delay: ', min_local_computation_delay, ' max local delay: ', max_local_computation_delay)
        #self.achieved_local_energy_consumption = interp(self.achieved_local_energy_consumption,[min_local_energy_consumption,max_local_energy_consumption],[0,5000])
        self.achieved_local_processing_delay = 1#interp(self.achieved_local_processing_delay,[min_local_computation_delay,max_local_computation_delay],[0,500])
        #print('')

    def offloading(self,communication_channel):
        offloading_bits = 0
        counter = 0
        self.dequeued_offload_tasks.clear()
        if self.achieved_channel_rate == 0:
            self.achieved_transmission_delay = 0
        else:
            left_bits = communication_channel.long_TTI*self.achieved_channel_rate
            for offloading_task in self.communication_queue:
                if offloading_task.slot_task_size < left_bits:
                    offloading_bits += offloading_task.slot_task_size
                    left_bits -= offloading_task.slot_task_size
                    self.dequeued_offload_tasks.append(offloading_task)
                    counter+=1

                elif offloading_task.slot_task_size > left_bits:
                    offloading_task.split_task(left_bits)
                    offloading_bits+=left_bits

            for x in range(0,counter):
                self.communication_queue.pop(0)
            self.achieved_transmission_delay = 1#self.packet_offload_size_bits/self.achieved_channel_rate

        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        achieved_throughput = []
        number_of_allocated_RBs = []
        total_size_bits_offloaded = []
        task_sizes = []

        if len(self.dequeued_offload_tasks) > 0:
            for dequeued_offload_task in self.dequeued_offload_tasks:
                task_identities.append(dequeued_offload_task.task_identifier)
                task_latency_requirements.append(dequeued_offload_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(dequeued_offload_task.queue_timer)
                achieved_throughput.append(self.achieved_channel_rate)
                number_of_allocated_RBs.append(len(self.allocated_RBs))
                task_sizes.append(dequeued_offload_task.slot_task_size)
                

            for dequeued_offload_task in self.dequeued_offload_tasks:    
                total_size_bits_offloaded.append(sum(task_sizes))

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency,
                "Number of allocated RBs": number_of_allocated_RBs,
                "Attained Throughput": achieved_throughput,
                "Offloaded Task Size":task_sizes,
                "Sum size of all offlaoded tasks": total_size_bits_offloaded
            }

            df = pd.DataFrame(data=data)

            #print('Dequeued Offload Tasks')
            #print(df)
            #print(' ')
            #print('Achieved TTI channel rate: ', self.achieved_channel_rate)
            #print(' ')

        self.check_completed_tasks()
            
        self.achieved_transmission_energy_consumption = self.assigned_transmit_power_W*self.achieved_transmission_delay
        #print('self.achieved_transmission_energy_consumption: ', self.achieved_transmission_energy_consumption)
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[0,12*math.pow(10,-5)],[0,100])
        #print('transmission energy consumed: ', self.achieved_transmission_energy_consumption)
        #min_offload_energy_consumption, max_offload_energy_consumption = self.min_and_max_achievable_offload_energy_consumption(communication_channel)
        #min_offloading_delay, max_offloading_delay = self.min_max_achievable_offload_delay(communication_channel)
        #print('min offload delay: ', min_offloading_delay, ' max offload delay: ', max_offloading_delay)
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[min_offload_energy_consumption,max_offload_energy_consumption],[0,5000])
        self.achieved_transmission_delay = 1#interp(self.achieved_transmission_delay,[min_offloading_delay,max_offloading_delay],[0,5000])
        #print('offload delay: ', self.achieved_transmission_delay)
        #print('transmission energy consumed: ', self.achieved_transmission_energy_consumption)
    
    def check_completed_tasks(self):
        local_queue_task_identities = []
        offload_queue_task_identities = []
        dequeued_offload_task_identities = []
        self.completed_tasks.clear()

        for dequeued_offload_task in self.dequeued_offload_tasks:
            dequeued_offload_task_identities.append(dequeued_offload_task.task_identifier)
        # First collect task identities of tasks which are currently in the local and offload queues
        for local_queue_task in self.local_queue:
            local_queue_task_identities.append(local_queue_task.task_identifier)

        for offload_queue_task in self.communication_queue:
            offload_queue_task_identities.append(offload_queue_task.task_identifier)

        # Find tasks dequeued from the local queue which are not present in the offloading queue (completed on local)
        for local_dequeued_task in self.dequeued_local_tasks:
            if local_dequeued_task.task_identifier not in offload_queue_task_identities and local_dequeued_task.task_identifier not in dequeued_offload_task_identities:
                self.completed_tasks.append(local_dequeued_task)

        for offload_dequeued_task in self.dequeued_offload_tasks:
            if offload_dequeued_task.task_identifier not in local_queue_task_identities:
                self.completed_tasks.append(offload_dequeued_task) 

        task_identities = []
        task_latency_requirements = []
        task_attained_queueing_latency = []
        if len(self.completed_tasks) > 0:
            for completed_task in self.completed_tasks:
                task_identities.append(completed_task.task_identifier)
                task_latency_requirements.append(completed_task.QOS_requirement.max_allowable_latency)
                task_attained_queueing_latency.append(completed_task.queue_timer)              

            data = {
                "Task Identity" : task_identities,
                "Latency Requirement" : task_latency_requirements,
                "Attained Queue Latency" : task_attained_queueing_latency
            }


            df = pd.DataFrame(data=data)

            #print('Completed Tasks')
            #print(df)
            #print(' ')
            #print(' ')

        sum_latency = 0
        for completed_task in self.completed_tasks:
            sum_latency+= (completed_task.QOS_requirement.max_allowable_latency - completed_task.queue_timer)

        self.queuing_delay = sum_latency
        

    def total_energy_consumed(self):
        if self.battery_energy_level >  self.achieved_total_energy_consumption:
            self.achieved_total_energy_consumption = self.achieved_local_energy_consumption + self.achieved_transmission_energy_consumption
            self.achieved_total_energy_consumption_normalized = interp(self.achieved_total_energy_consumption,[0,1000],[0,1])
            self.battery_energy_level = self.battery_energy_level - self.achieved_total_energy_consumption
        else:
            self.achieved_total_energy_consumption = 0

        #print(self.battery_energy_level)
        #print('total energy: ', self.achieved_total_energy_consumption)

    def total_processing_delay(self):
        self.achieved_total_processing_delay = self.achieved_local_processing_delay + self.achieved_transmission_delay
        #print('eMBB User: ', self.eMBB_UE_label, 'achieved delay: ', self.achieved_total_processing_delay)
        #print(' ')
        #print('offload ratio: ', self.allocated_offloading_ratio, 'local delay: ', self.achieved_local_processing_delay, 'offlaod delay: ', self.achieved_transmission_delay)
        
    
    def calculate_channel_gain(self):
        #Pathloss gain
        self.pathloss_gain = (math.pow(10,(35.3+37.6*math.log10(self.distance_from_SBS))))/10
        self.small_scale_channel_gain = np.random.exponential(1)
        #self.large_scale_channel_gain = np.random.lognormal(0.0,1.0)
        self.total_gain = self.small_scale_channel_gain#*self.large_scale_channel_gain#self.pathloss_gain
        #if self.total_gain < 0.1:
        #    self.total_gain = 0.1

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

    def min_and_max_achievable_rates(self, communication_channel):
        min_num_RB = communication_channel.num_allocate_RBs_lower_bound
        max_num_RB = communication_channel.num_allocate_RBs_upper_bound
        RB_bandwidth_Hz = communication_channel.RB_bandwidth_Hz
        min_channel_rate_numerator = self.min_channel_gain*self.min_transmission_power_W
        max_channel_rate_numerator = self.max_channel_gain*self.max_transmission_power_W
        channel_rate_denominator = communication_channel.noise_spectral_density_W*RB_bandwidth_Hz
        min_achievable_rate = min_num_RB*(RB_bandwidth_Hz*math.log2(1+(min_channel_rate_numerator/channel_rate_denominator)))
        max_achievable_rate = max_num_RB*(RB_bandwidth_Hz*math.log2(1+(max_channel_rate_numerator/channel_rate_denominator)))

        return min_achievable_rate, max_achievable_rate
    
    def min_and_max_achievable_local_energy_consumption(self):
        #Local Consumption
        cycles_per_bit_max = self.cpu_cycles_per_byte*8*(5000*8000)
        achieved_local_energy_consumption_max = self.energy_consumption_coefficient*math.pow(self.cpu_clock_frequency,2)*cycles_per_bit_max
        achieved_local_energy_consumption_min = 0
        #print("max achievable local energy consumption: ",achieved_local_energy_consumption_max)
        return achieved_local_energy_consumption_min, achieved_local_energy_consumption_max
    
    def min_max_achievable_local_processing_delay(self):
        cycles_per_bit_max = self.cpu_cycles_per_byte*8*(5000*8000)
        cycles_per_bit_min = self.cpu_cycles_per_byte*8*(1000*8000)
        achieved_local_processing_delay_max = cycles_per_bit_max/self.cpu_clock_frequency
        achieved_local_processing_delay_min = cycles_per_bit_min/self.cpu_clock_frequency
        return achieved_local_processing_delay_min, achieved_local_processing_delay_max
    
    def min_and_max_achievable_offload_energy_consumption(self,communication_channel):
        #Offloading energy
        min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        max_achieved_transmission_delay = (5000*8000)/min_achievable_rate
        achieved_transmission_energy_consumption_max = self.max_transmission_power_W*max_achieved_transmission_delay
        #self.achieved_transmission_energy_consumption = interp(self.achieved_transmission_energy_consumption,[0,12*math.pow(10,-5)],[0,100])
        achieved_transmission_energy_consumption_min = 0
        #print("max achievable offloading energy consumption: ", achieved_transmission_energy_consumption_max)
        return achieved_transmission_energy_consumption_min, achieved_transmission_energy_consumption_max
    
    def min_max_achievable_offload_delay(self,communication_channel):
        min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        max_achieved_transmission_delay = (5000*8000)/min_achievable_rate
        min_achieved_transmission_delay = (1000*8000)/max_achievable_rate
        return min_achieved_transmission_delay, max_achieved_transmission_delay
    
    def calculate_delay_penalty(self):
        
        if (self.allowable_latency - self.achieved_total_processing_delay) >= 0:
            delay_reward = self.delay_reward
        else:
            delay_reward = (self.allowable_latency - self.achieved_total_processing_delay)
        #print('self.allowable_latency: ', self.allowable_latency)
        #print('self.achieved_total_processing_delay: ', self.achieved_total_processing_delay)
        #print('self.allowable_latency - self.achieved_total_processing_delay: ', self.allowable_latency - self.achieved_total_processing_delay)
        delay_reward = self.allowable_latency - self.achieved_total_processing_delay
        min_delay = -2200
        max_delay = 1980
        delay_reward = interp(delay_reward,[min_delay,max_delay],[0,1])
        return delay_reward
    
    def calculate_energy_efficiency(self):
        if self.achieved_total_energy_consumption == 0:
            energy_efficiency = 0
        else:
            energy_efficiency = self.achieved_channel_rate/self.achieved_total_energy_consumption #0.4*self.achieved_channel_rate_normalized/0.6*self.achieved_total_energy_consumption_normalized 
            #energy_efficiency = self.achieved_total_energy_consumption_normalized 
            
        min_energy_efficiency = 0
        max_energy_efficiency = 500
        energy_efficiency = interp(energy_efficiency,[min_energy_efficiency,max_energy_efficiency],[0,1])
        return energy_efficiency
    
    def calculate_throughput_reward(self,communication_channel):
        queue_size = self.user_state_space.calculate_communication_queue_size()
        #normalize queue size
        #queue_size_normalized = interp(queue_size,[0,self.max_communication_qeueu_size],[0,1])
        #print('queue size: ', queue_size)

        #normalize achieved thoughput
        #min_achievable_rate, max_achievable_rate = self.min_and_max_achievable_rates(communication_channel)
        #achieved_channel_rate_normalized = interp(self.achieved_channel_rate,[min_achievable_rate,max_achievable_rate],[0,1])
        throughput_reward = self.achieved_channel_rate - queue_size
        #print('throughput reward: ', throughput_reward)
        #Normailze throughput reward
        min_throughput_reward = -28960000
        max_throughput_reward = 159844000
        if(throughput_reward > 0):
            throughput_reward_normalized = interp(throughput_reward,[min_throughput_reward,max_throughput_reward],[0,1])
        else:
            throughput_reward_normalized = -0.65
        return throughput_reward_normalized

    def compute_battery_energy_level(self):
        self.previous_slot_battery_energy = self.battery_energy_level
        self.battery_energy_level = self.battery_energy_level + self.energy_harvested

    def harvest_energy(self):
        self.energy_harvested = np.random.exponential(250)#random.randint(0,2000)

    def energy_consumption_reward(self):
        energy_reward = self.battery_energy_level + self.energy_harversted - self.achieved_total_energy_consumption

        max_energy_reward = 22000
        min_energy_reward = 0

        if energy_reward >= 0:
            energy_reward_normalized = interp(energy_reward,[min_energy_reward,max_energy_reward],[0,1])
        else:
            energy_reward_normalized = -0.2

        return energy_reward_normalized
    
    def increment_task_queue_timers(self):
        if len(self.task_queue) > 0:
            for task in self.task_queue:
                task.increment_queue_timer()

        if len(self.local_queue) > 0:
            for local_task in self.local_queue:
                local_task.increment_queue_timer()

        if len(self.communication_queue) > 0:
            for offload_task in self.communication_queue:
                offload_task.increment_queue_timer()

    def queueing_delay_reward(self):
        if self.queuing_delay > 0:
            qeueuing_delay_reward = 1
        else:
            qeueuing_delay_reward = self.queuing_delay

        return qeueuing_delay_reward
        

        






  



            