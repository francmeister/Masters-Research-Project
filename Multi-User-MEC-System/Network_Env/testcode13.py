inf_total_reward = []
inf_energy = []
inf_task_delays = []
inf_throughput = []
inf_fairness_index = []
inf_num_RBs_allocated = []
inf_outage_probability = []
inf_individual_channel_rates = []
inf_individual_number_of_allocated_RB = []
inf_individual_number_of_puncturing_urllc_users = []
inf_individual_num_of_clustered_urllc_users = []
inf_failed_urllc_transmissions = []
inf_individual_offload_ratios = []
inf_individual_local_queue_lengths_bits = []
inf_individual_offload_queue_lengths_bits = []
inf_individual_local_queue_lengths_tasks = []
inf_individual_offload_queue_lengths_tasks = []

inf_battery_energy_constraint_violation_count = []
inf_local_queueing_traffic_constraint_violation_count = []
inf_offload_queueing_traffic_constaint_violation_count = []
inf_local_time_delay_violation_prob_constraint_violation_count = []
inf_offload_time_delay_violation_prob_constraint_violation_count = []
inf_rmin_constraint_violation_count = []
inf_individual_energy_harvested_levels = []
inf_individual_battery_energy_levels = []
inf_individual_local_queue_delay_violation_probability = []
inf_individual_offload_queue_delay_violation_probability = []
inf_total_local_delay = []
inf_total_offload_delay = []
inf_total_local_queue_length_tasks = []
inf_total_offload_queue_length_tasks = []
inf_total_local_queue_length_bits = []
inf_total_offload_queue_length_bits = []

inf_total_urllc_data_rate = []

inf_individual_urllc_data_rate = []
inf_number_of_arriving_urllc_packets = []
inf_number_of_dropped_urllc_packets_due_to_resource_allocation = []
inf_number_of_dropped_urllc_packets_due_to_channel_rate = []
inf_urllc_successful_transmissions = []
inf_individual_number_of_arriving_urllc_packets = []
inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation = []
inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate = []
inf_individual_successful_transmissions = []
inf_cb_allocations_count = []
inf_L_values = []
inf_cdf_values = []

def evaluate_policy(policy, eval_episodes,number_of_users):
  inf_outage_probability=[]
  avg_reward = 0
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(obs)
      noise = np.random.normal(0, expl_noise, size=env.action_space_dim)
      action = (action + noise).clip(env.action_space_low, env.action_space_high)
      action = env.reshape_action_space_from_model_to_dict(action)
      mode = 'inference'
      reformed_action = env.apply_resource_allocation_constraint(action,mode)
      obs, reward, done, _ = env.step(reformed_action)
      inf_throughput.append(env.total_rate)#
      inf_fairness_index.append(env.SBS1.fairness_index)#
      inf_outage_probability.append(env.SBS1.outage_probability)#
      inf_failed_urllc_transmissions.append(env.SBS1.failed_urllc_transmissions)#
      inf_individual_offload_ratios.append(env.offload_decisions)#
      inf_total_urllc_data_rate.append(env.SBS1.urllc_total_rate_per_second)#
      inf_number_of_arriving_urllc_packets.append(env.SBS1.number_of_arriving_urllc_packets)#
      inf_number_of_dropped_urllc_packets_due_to_resource_allocation.append(env.SBS1.number_of_dropped_urllc_packets_due_to_resource_allocation)#
      inf_number_of_dropped_urllc_packets_due_to_channel_rate.append(env.SBS1.number_of_dropped_urllc_packets_due_to_channel_rate)#
      inf_urllc_successful_transmissions.append(env.SBS1.urllc_successful_transmissions)#
      inf_individual_successful_transmissions.append(env.SBS1.individual_successful_transmissions)
      inf_individual_urllc_data_rate.append(env.SBS1.individual_urllc_channel_rate_per_second_with_penalty)
      inf_individual_number_of_arriving_urllc_packets.append(env.SBS1.individual_number_of_arriving_urllc_packets)
      inf_individual_number_of_dropped_urllc_packets_due_to_resource_allocation.append(env.SBS1.individual_number_of_dropped_urllc_packets_due_to_resource_allocation)
      inf_individual_number_of_dropped_urllc_packets_due_to_channel_rate.append(env.SBS1.individual_number_of_dropped_urllc_packets_due_to_channel_rate)
      inf_cb_allocations_count.append(env.SBS1.cb_allocations_count)#

      avg_reward += reward
  avg_reward /= eval_episodes

  av_throughput = sum(inf_throughput)/len(inf_throughput)#throughput_policy__multiplexing
  av_fairness_index = sum(inf_fairness_index)/len(inf_fairness_index)#fairness_index_policy__multiplexing
  inf_outage_probability = [0 if math.isnan(x) else x for x in inf_outage_probability]
  av_outage_probability = sum(inf_outage_probability)/len(inf_outage_probability)#outage_probability_policy__multiplexing
  av_total_urllc_data_rate = sum(inf_total_urllc_data_rate)/len(inf_total_urllc_data_rate)#urllc_throughput_policy__multiplexing
  av_individual_offload_ratios = np.array(inf_individual_offload_ratios)
  av_individual_offload_ratios = np.mean(av_individual_offload_ratios, axis=0)
  av_individual_offload_ratios = np.mean(av_individual_offload_ratios)#offloading_ratios_policy__multiplexing
  av_inf_failed_urllc_transmissions = sum(inf_failed_urllc_transmissions)/len(inf_failed_urllc_transmissions)
  av_number_of_arriving_urllc_packets = sum(inf_number_of_arriving_urllc_packets)/len(inf_number_of_arriving_urllc_packets)#urllc_arriving_packets_policy__multiplexing
  av_number_of_dropped_urllc_packets_due_to_resource_allocation = sum(inf_number_of_dropped_urllc_packets_due_to_resource_allocation)/len(inf_number_of_dropped_urllc_packets_due_to_resource_allocation)
  av_number_of_dropped_urllc_packets_due_to_channel_rate = sum(inf_number_of_dropped_urllc_packets_due_to_channel_rate)/len(inf_number_of_dropped_urllc_packets_due_to_channel_rate)
  av_urllc_successful_transmissions = sum(inf_urllc_successful_transmissions)/len(inf_urllc_successful_transmissions)
  av_inf_failed_urllc_transmissions = av_inf_failed_urllc_transmissions/av_number_of_arriving_urllc_packets#failed_urllc_transmissions_policy__multiplexing
  av_urllc_successful_transmissions = av_urllc_successful_transmissions/av_number_of_arriving_urllc_packets#urllc_successful_transmissions_policy__multiplexing
  av_number_of_dropped_urllc_packets_due_to_resource_allocation = av_number_of_dropped_urllc_packets_due_to_resource_allocation/av_number_of_arriving_urllc_packets#
  av_number_of_dropped_urllc_packets_due_to_channel_rate = av_number_of_dropped_urllc_packets_due_to_channel_rate/av_number_of_arriving_urllc_packets#urllc_dropped_packets_channel_rate_policy__multiplexing
  av_cb_allocations_count = sum(inf_cb_allocations_count)/len(inf_cb_allocations_count)#urllc_code_blocks_allocation_policy__multiplexing
  av_number_of_dropped_urllc_packets_due_to_channel_rate_ = av_number_of_dropped_urllc_packets_due_to_channel_rate/av_cb_allocations_count#urllc_dropped_packets_channel_rate_normalized_policy__multiplexing
  np.set_printoptions(threshold=np.inf)



# offloading_ratios_policy_3_multiplexing 

  print('')
  print ("---------------------------------------")
  print ("Average Throughput over the Evaluation Step: %f" % (av_throughput))
  print ("---------------------------------------")
  print('')
  print ("---------------------------------------")
  print ("Average Fairness Index over the Evaluation Step: %f" % (av_fairness_index))
  print ("---------------------------------------")
  print('')

  print ("Average Outage Probability over the Evaluation Step: %f" % (av_outage_probability))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Failed URLLC Transmissions over the Evaluation Step: %f" % (av_inf_failed_urllc_transmissions))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average URLLC Throughput over the Evaluation Step: %f" % (av_total_urllc_data_rate))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Number of arriving URLLC packets: %f" % (av_number_of_arriving_urllc_packets))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Number of Dropped Packets (Resource Allocation): %f" % (av_number_of_dropped_urllc_packets_due_to_resource_allocation))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Number of Dropped Packets (Channel Rate): %f" % (av_number_of_dropped_urllc_packets_due_to_channel_rate))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Number of Dropped Packets (Channel Rate) Normalized: %f" % (av_number_of_dropped_urllc_packets_due_to_channel_rate_))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Number of Successful Transmissions: %f" % (av_urllc_successful_transmissions))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Number of Code Block allocations: %f" % (av_cb_allocations_count))
  print ("---------------------------------------")
  print ("---------------------------------------")
  print ("Average Offloading Ratios: ",av_individual_offload_ratios)
  print ("---------------------------------------")
