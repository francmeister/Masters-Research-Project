import pygame, sys, time, random
pygame.init()
import math
import numpy as np
from numpy import interp
import scipy.stats as stats
import statistics

class SBS():
    def __init__(self, SBS_label):
        self.SBS_label = SBS_label
        #SBS Telecom properties
        self.x_position = 200
        self.y_position = 200
        self.individual_rewards = []
        self.set_properties()

    def associate_users(self, eMBB_Users, URLLC_users):
        self.associated_eMBB_users = eMBB_Users
        self.associated_URLLC_users = URLLC_users

    def get_SBS_center_pos(self):
        self.x_position = 200
        self.y_position = 200

    def collect_state_space(self, eMBB_Users,urllc_users):

    
        self.system_state_space_RB_channel_gains.clear()
        self.system_state_space_battery_energies.clear()
        channel_gains = []
        communication_queue_size = []
        battery_energy = []
        offloading_queue_lengths = []
        local_queue_lengths = []
        num_arriving_urllc_packets = []
        latency_requirement = []
        local_frequencies = []
        #reliability_requirement = []
        #Collect Channel gains
        self.count_num_arriving_urllc_packets(urllc_users)
        for embb_user in eMBB_Users:
            channel_gains.append(embb_user.user_state_space.channel_gain)
            #communication_queue_size.append(user.user_state_space.calculate_communication_queue_size())
            battery_energy.append(embb_user.user_state_space.battery_energy)
            offloading_queue_lengths.append(embb_user.user_state_space.offloading_queue_length)
            local_queue_lengths.append(embb_user.user_state_space.local_queue_length)
            num_arriving_urllc_packets.append(self.num_arriving_urllc_packets)
            #latency_requirement.append(0)
            #latency_requirement.append(user.user_state_space.QOS_requirements.max_allowable_latency)
            #local_frequencies.append(user.user_state_space.local_cpu_frequency)
            #reliability_requirement.append(user.user_state_space.QOS_requirements.max_allowable_reliability)
        #print('state space')
        #print(channel_gains[0])
        #print(battery_energy)
        self.system_state_space_RB_channel_gains.append(channel_gains)
        #self.system_state_space.append(communication_queue_size)
        self.system_state_space_battery_energies.append(battery_energy)
        return channel_gains, battery_energy, offloading_queue_lengths, local_queue_lengths, num_arriving_urllc_packets
        #return channel_gains, battery_energy

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

    def receive_offload_packets(self, eMBB_Users):
        pass
        #for eMBB_User in eMBB_Users:
            #if eMBB_User.has_transmitted_this_time_slot == True:
                #self.eMBB_Users_packet_queue.append(eMBB_User.offloaded_packet)

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

    def calculate_achieved_system_energy_efficiency(self):
        if self.achieved_total_system_energy_consumption == 0:
            self.achieved_system_energy_efficiency = 0
        else:
            self.achieved_system_energy_efficiency = self.achieved_total_rate_eMBB_users/self.achieved_total_system_energy_consumption

        #print("self.achieved_system_energy_efficiency",self.achieved_system_energy_efficiency)

    def calculate_achieved_system_reward(self, eMBB_Users, urllc_users, communication_channel, q_action):
        #print('number of embb users: ', len(eMBB_Users))
        self.achieved_system_reward = 0
        eMBB_User_energy_consumption = 0
        eMBB_User_channel_rate = 0
        eMBB_User_QOS_requirement_revenue_or_penelaty = 0
        total_energy = 0
        total_rate = 0
        total_QOS_revenue = 0
        self.fairness_index = 0
        individual_rewards = 0
        queue_delay_reward = 0
        delay = 0
        throughput_reward = 0
        resource_allocation_reward = 0
        tasks_dropped = 0

        self.individual_rewards.clear()
        self.individual_energy_rewards = []
        self.individual_channel_rate_rewards = []
        self.individual_channel_battery_energy_rewards = []
        self.individual_delay_rewards = []
        self.individual_queue_delays = []
        self.individual_tasks_dropped = []
        self.individual_energy_efficiency = []
        self.individual_total_reward = []
        total_users_energy_reward = 0
        total_users_throughput_reward = 0
        total_users_battery_energies_reward = 0
        total_users_delay_rewards = 0
        total_users_delay_times_energy_reward = 0
        total_users_resource_allocation_reward = 0
        overall_users_reward = 0
        total_eMBB_User_delay_normalized = 0
        total_offload_traffic_reward = 0
        total_lc_delay_violation_probability = 0
        urllc_reliability_reward, urllc_reliability_reward_normalized = self.calculate_urllc_reliability_reward(urllc_users)
        users_channel_rates = []
        #d
        self.q_action = q_action[0]
        self.urllc_reliability_reward_normalized = urllc_reliability_reward_normalized
        for eMBB_User in eMBB_Users:
            eMBB_User_delay, eMBB_User_delay_normalized = eMBB_User.new_time_delay_calculation()
            eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption_normalized 
            #eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption
            total_energy += eMBB_User_energy_consumption
            eMBB_User_channel_rate = eMBB_User.achieved_channel_rate_normalized
            #eMBB_User_channel_rate = eMBB_User.achieved_channel_rate
            users_channel_rates.append(eMBB_User_channel_rate)
            #eMBB_User_channel_rate = eMBB_User.achieved_channel_rate
            total_rate += eMBB_User_channel_rate
            delay_reward = eMBB_User.calculate_delay_penalty()
            battery_energy_reward = eMBB_User.energy_consumption_reward()
            energy_efficiency_reward = eMBB_User.calculate_energy_efficiency()
            resource_allocation_reward = eMBB_User.calculate_resource_allocation_reward(communication_channel)
            queue_delay_reward,delay = eMBB_User.calculate_queuing_delays()
            tasks_dropped = eMBB_User.tasks_dropped
            total_offload_traffic_reward += eMBB_User.offloading_queue_stability_constraint_reward()
            #total_lc_delay_violation_probability+=eMBB_User.local_queue_violation_constraint_reward()

            total_eMBB_User_delay_normalized+=eMBB_User_delay_normalized
            total_users_energy_reward += eMBB_User_energy_consumption
            total_users_throughput_reward += eMBB_User_channel_rate
            total_users_battery_energies_reward += battery_energy_reward
            total_users_delay_rewards += queue_delay_reward
            if eMBB_User_energy_consumption == 0:
                total_users_delay_times_energy_reward = 0
            else:
                total_users_delay_times_energy_reward += (queue_delay_reward*(1/eMBB_User_energy_consumption))
            total_users_resource_allocation_reward += resource_allocation_reward

        
            #if eMBB_User_energy_consumption == 0:
            #    individual_reward = 0
            #else:
            individual_reward = energy_efficiency_reward#*queue_delay_reward + battery_energy_reward  
      
            self.achieved_system_reward += individual_reward
            self.individual_rewards.append(individual_reward)

            self.energy_efficiency_rewards+=energy_efficiency_reward
            self.battery_energy_rewards+=battery_energy_reward
            self.throughput_rewards+=total_rate
            self.energy_rewards+=total_energy
            self.delay_rewards+=queue_delay_reward
            self.delays+=delay
            self.tasks_dropped+=tasks_dropped
            self.resource_allocation_rewards += resource_allocation_reward
            self.individual_energy_rewards.append(eMBB_User_energy_consumption)
            self.individual_channel_rate_rewards.append(eMBB_User_channel_rate)
            self.individual_energy_efficiency.append(energy_efficiency_reward)
            self.individual_total_reward.append(individual_reward)
            self.individual_channel_battery_energy_rewards.append(battery_energy_reward)
            self.individual_tasks_dropped.append(tasks_dropped)
            self.individual_delay_rewards.append(queue_delay_reward)
            self.individual_queue_delays.append(delay)
            self.total_reward += energy_efficiency_reward*queue_delay_reward + battery_energy_reward

        #self.overall_users_reward = total_users_throughput_reward*total_users_delay_times_energy_reward + total_users_battery_energies_reward
        if self.energy_rewards > 0 and self.throughput_rewards > 0:
            self.overall_users_reward = self.throughput_rewards - self.q_action*self.energy_rewards
        else:
            self.overall_users_reward = 0
        #print('overall_users_reward: ', self.overall_users_reward)
        #overall_users_rewards = [overall_users_reward for _ in range(len(eMBB_Users))]
        #self.achieved_system_reward += urllc_reliability_reward_normalized
        fairness_index = self.calculate_fairness(eMBB_Users)
        #print('fairness index: ', fairness_index)
        fairness_index_normalized = interp(fairness_index,[0,1],[0,3])
        #print('fairness index: ', fairness_index_normalized)
        #print(' ')
        #fairness_penalty = self.calculate_fairness_(eMBB_Users, communication_channel)
        #print('fairness penalty: ', fairness_penalty)
        #print('fairness penalty: ', fairness_index_normalized)
        #print(' ')
        self.fairness_index = fairness_index
        new_individual_rewards = [x + fairness_index_normalized for x in self.individual_rewards]
        self.users_rate_variance = statistics.pvariance(users_channel_rates)
        self.users_rate_variance_sum+=statistics.pvariance(users_channel_rates)
      
        self.achieved_system_reward = self.achieved_system_reward + fairness_index_normalized
        #self.achieved_system_reward = fairness_index_normalized
        #return self.achieved_system_reward, urllc_reliability_reward_normalized, self.energy_rewards,self.throughput_rewards
        #print('self.achieved_system_reward: ', self.achieved_system_reward)
        return self.achieved_system_reward, self.achieved_system_reward, self.energy_rewards,self.throughput_rewards
        #return self.achieved_system_reward, self.overall_users_reward , self.energy_rewards,self.throughput_rewards

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
        self.users_rate_variance = 0
        self.users_rate_variance_sum = 0
        self.q_action = 0
        self.associated_users = []
        self.associated_URLLC_users = []
        self.associated_eMBB_users = []
        self.system_state_space_RB_channel_gains = []
        self.system_state_space_battery_energies = []
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
        self.fairness_index = 0
        self.energy_efficiency_rewards = 0
        self.energy_rewards = 0
        self.throughput_rewards = 0
        self.delay_rewards = 0
        self.battery_energy_rewards = 0
        self.delays = 0
        self.tasks_dropped = 0
        self.resource_allocation_rewards = 0
        self.delay_reward_times_energy_reward = 0
        self.available_resource_time_blocks = []
        self.num_arriving_urllc_packets = 0
        self.urllc_reliability_constraint_max = 0.04
        self.K_mean = 0
        self.K_variance = 3
        self.outage_probability = 0
        self.previous_rates = []
        self.previous_Ks = []
        self.timeslot_counter = 0
        self.ptr = 0
        self.Kptr = 0
        self.urllc_reliability_reward_normalized = 0
        self.q = 0
        self.individual_energy_rewards = []
        self.individual_channel_rate_rewards = []
        self.individual_channel_battery_energy_rewards = []
        self.individual_delay_rewards = []
        self.individual_queue_delays = []
        self.individual_tasks_dropped = []
        self.individual_energy_efficiency = []
        self.individual_total_reward = []
        self.total_reward = 0
        self.overall_users_reward = 0

    def calculate_fairness(self,eMBB_Users):
        number_of_users = len(eMBB_Users)
        sum_throughputs = 0
        for eMBB_User in eMBB_Users:
            sum_throughputs += eMBB_User.achieved_channel_rate

        square_sum_throughput = math.pow(sum_throughputs,2)
        sum_square_throughput = self.square_sum(eMBB_Users)
        fairness_index = 0
        if sum_square_throughput > 0:
            fairness_index = square_sum_throughput/(number_of_users*sum_square_throughput)

        return fairness_index

    def square_sum(self,eMBB_Users):
        sum = 0
        for eMBB_User in eMBB_Users:
            sum+=math.pow(eMBB_User.achieved_channel_rate,2)

        return sum
    
    def calculate_fairness_(self,eMBB_Users, communication_channel):
        sum_square_error = 0
        for eMBB_user in eMBB_Users:
            square_error = math.pow(abs(len(eMBB_user.allocated_RBs)-communication_channel.num_of_RBs_per_User),2)
            sum_square_error+=square_error

        sum_square_error = math.pow(abs((len(eMBB_Users[0].allocated_RBs))-(len(eMBB_Users[1].allocated_RBs))),2)
        if sum_square_error == 0:
            sum_square_error = 1
        
        return 1/sum_square_error
    
    def allocate_resource_blocks_URLLC(self,communication_channel, URLLC_Users):
        for URLLC_user in URLLC_Users:
            URLLC_user.calculate_channel_gain_on_all_resource_blocks(communication_channel)

        for rb in range(1,communication_channel.num_allocate_RBs_upper_bound+1):
            for tb in range(1,communication_channel.time_divisions_per_slot+1):
                self.available_resource_time_blocks.append((tb,rb))
        
        for urllc_user in URLLC_Users:
            random_number = np.random.randint(0, len(self.available_resource_time_blocks), 1)
            random_number = random_number[0]
            urllc_user.assigned_resource_time_block = self.available_resource_time_blocks[random_number]
            self.available_resource_time_blocks = np.delete(self.available_resource_time_blocks,random_number,axis=0)
            urllc_user.assigned_time_block = urllc_user.assigned_resource_time_block[0]
            urllc_user.assigned_resource_block = urllc_user.assigned_resource_time_block[1]

    def count_num_arriving_urllc_packets(self, urllc_users):
        self.num_arriving_urllc_packets = 0
        for urllc_user in urllc_users:
            if urllc_user.has_transmitted_this_time_slot == True:
                self.num_arriving_urllc_packets += 1


        

        

        # for urllc_user in URLLC_Users:
        #     print('urllc user: ', urllc_user.URLLC_UE_label, 'allocated resource block id: ', urllc_user.assigned_resource_block)
        #     print('urllc user: ', urllc_user.URLLC_UE_label, 'allocated time block id: ', urllc_user.assigned_time_block)

        # print('')

        


        



        # URLLC_Users_temp = URLLC_Users
        # RB_URLLC_mapping = []
        # for x in range(1,communication_channel.num_allocate_RBs_upper_bound+1):
        #     for y in range(0,communication_channel.num_urllc_users_per_RB):
        #         if len(URLLC_Users_temp) > 0:

    def get_top_two_indices(arr):
        # Get the indices of the array sorted in ascending order
        sorted_indices = np.argsort(arr)

        # Get the index of the largest number (last index after sorting)
        largest_index = sorted_indices[-1]

        # Get the index of the second largest number (second-to-last index after sorting)
        second_largest_index = sorted_indices[-2]

        return largest_index
    
    def calculate_urllc_reliability_reward(self, urllc_users):
        num_arriving_urllc_packets = self.num_arriving_urllc_packets
        urllc_task_size = 0
        if len(urllc_users) > 0:
            urllc_task_size = urllc_users[0].task_size_per_slot_bits    

        urllc_total_rate = 0
        rates = []
        for urllc_user in urllc_users:
            urllc_total_rate+=urllc_user.achieved_channel_rate
            rates.append(urllc_user.achieved_channel_rate)

      
       
        K = num_arriving_urllc_packets*urllc_task_size
        K_mean = self.K_expectation_over_prev_T_slot(10,K)
        if len(urllc_users) > 3:
            K_variance = (len(urllc_users)-2)*urllc_task_size
        else:
            K_variance = (1)*urllc_task_size
        #K_inv = stats.norm.ppf(K, loc=K_mean, scale=K_variance)
     


        reliability_reward = urllc_total_rate-K*(1-self.urllc_reliability_constraint_max)
        if reliability_reward < 0:
            reliability_reward = 0
        else:
            reliability_reward = reliability_reward
        #average_rate_prev_slots = self.urllc_rate_expectation_over_prev_T_slot(10,urllc_total_rate)
        average_rate = urllc_total_rate/len(urllc_users)
        variance_rate = statistics.pvariance(rates)
        std_rate = math.sqrt(variance_rate)
     
        #print('self.previous_rates: ', self.previous_rates)
        #variance = urllc_task_size

        self.outage_probability = stats.norm.cdf(K,loc=average_rate,scale=std_rate)
        # print('reliability_reward: ', reliability_reward)
        # print('self.outage_probability: ', self.outage_probability)
        #print('reliability_reward: ', reliability_reward)
        reliability_reward_max = 4000
        reliability_reward_min = 0
        reliability_reward_normalized = interp(reliability_reward,[reliability_reward_min,reliability_reward_max],[0,5])
        return reliability_reward, reliability_reward_normalized
    
    def urllc_rate_expectation_over_prev_T_slot(self, T, urllc_total_rate):
        self.timeslot_counter+=1
        number_of_previous_time_slots = T

        if len(self.previous_rates) == number_of_previous_time_slots:
            self.previous_rates[int(self.ptr)] = urllc_total_rate
            self.ptr = (self.ptr + 1) % number_of_previous_time_slots
        else:
            self.previous_rates.append(urllc_total_rate)

        average_rate = sum(self.previous_rates)/len(self.previous_rates)
        return average_rate
    
    def K_expectation_over_prev_T_slot(self, T, K):
        self.timeslot_counter+=1
        number_of_previous_time_slots = T

        if len(self.previous_Ks) == number_of_previous_time_slots:
            self.previous_Ks[int(self.Kptr)] = K
            self.Kptr = (self.Kptr + 1) % number_of_previous_time_slots
        else:
            self.previous_Ks.append(K)

        average_K = sum(self.previous_Ks)/len(self.previous_Ks)
        return average_K
    

    def reward(self, eMBB_Users, urllc_users, communication_channel):
        #print('number of embb users: ', len(eMBB_Users))
        self.achieved_system_reward = 0
        eMBB_User_energy_consumption = 0
        eMBB_User_channel_rate = 0
        eMBB_User_QOS_requirement_revenue_or_penelaty = 0
        total_energy = 0
        total_rate = 0
        total_QOS_revenue = 0
        self.fairness_index = 0
        individual_rewards = 0
        queue_delay_reward = 0
        delay = 0
        throughput_reward = 0
        resource_allocation_reward = 0
        tasks_dropped = 0

        self.individual_rewards.clear()

        total_users_energy_reward = 0
        total_users_throughput_reward = 0
        total_users_battery_energies_reward = 0
        total_users_delay_rewards = 0
        total_users_delay_times_energy_reward = 0
        total_users_resource_allocation_reward = 0
        overall_users_reward = 0
        total_eMBB_User_delay_normalized = 0
        total_offload_traffic_reward = 0
        total_lc_delay_violation_probability = 0
        urllc_reliability_reward, urllc_reliability_reward_normalized = self.calculate_urllc_reliability_reward(urllc_users)
        self.urllc_reliability_reward_normalized = urllc_reliability_reward_normalized
        for eMBB_User in eMBB_Users:
            eMBB_User_delay, eMBB_User_delay_normalized = eMBB_User.new_time_delay_calculation()
            eMBB_User_energy_consumption = eMBB_User.achieved_total_energy_consumption 
            total_energy += eMBB_User_energy_consumption
            eMBB_User_channel_rate = eMBB_User.achieved_channel_rate
            total_rate += eMBB_User_channel_rate
            delay_reward = eMBB_User.calculate_delay_penalty()
            battery_energy_reward = eMBB_User.energy_consumption_reward()
            energy_efficiency_reward = eMBB_User.calculate_energy_efficiency()
            resource_allocation_reward = eMBB_User.calculate_resource_allocation_reward(communication_channel)
            queue_delay_reward,delay = eMBB_User.calculate_queuing_delays()
            tasks_dropped = eMBB_User.tasks_dropped
            total_offload_traffic_reward += eMBB_User.offloading_queue_stability_constraint_reward()
            total_lc_delay_violation_probability+=eMBB_User.local_queue_violation_constraint_reward()

            total_eMBB_User_delay_normalized+=eMBB_User_delay_normalized
            total_users_energy_reward += eMBB_User_energy_consumption
            total_users_throughput_reward += eMBB_User_channel_rate
            total_users_battery_energies_reward += battery_energy_reward
            total_users_delay_rewards += queue_delay_reward
            if eMBB_User_energy_consumption == 0:
                total_users_delay_times_energy_reward = 0
            else:
                total_users_delay_times_energy_reward += (queue_delay_reward*(1/eMBB_User_energy_consumption))
            total_users_resource_allocation_reward += resource_allocation_reward

        
            #if eMBB_User_energy_consumption == 0:
            #    individual_reward = 0
            #else:
            individual_reward = energy_efficiency_reward#*queue_delay_reward + battery_energy_reward  
      
            self.achieved_system_reward += individual_reward
            self.individual_rewards.append(individual_reward)

            self.energy_efficiency_rewards+=energy_efficiency_reward
            self.battery_energy_rewards+=battery_energy_reward
            self.throughput_rewards+=total_rate
            self.energy_rewards+=total_energy
            self.delay_rewards+=queue_delay_reward
            self.delays+=delay
            self.tasks_dropped+=tasks_dropped
            self.resource_allocation_rewards += resource_allocation_reward

        
        self.q = total_users_throughput_reward*total_users_delay_times_energy_reward 
        reward = total_users_throughput_reward - self.q*total_users_delay_times_energy_reward + total_users_battery_energies_reward + urllc_reliability_reward + total_offload_traffic_reward +total_lc_delay_violation_probability
        
        return self.achieved_system_reward, reward , self.energy_rewards,self.throughput_rewards






        




        







