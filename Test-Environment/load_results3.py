import numpy as np
import matplotlib.pyplot as plt

#a_load = np.load('TD3_NetworkEnv-v0_0.npy')

rewards_throughput_energy_1_users = np.load('TD3_NetworkEnv-v0_0_1_users.npy')
rewards_throughput_energy_3_users = np.load('TD3_NetworkEnv-v0_0_3_users.npy')
rewards_throughput_energy_5_users = np.load('TD3_NetworkEnv-v0_0_5_users.npy')
fairness_index = np.load('fairnes_index.npy')


timesteps_1 = rewards_throughput_energy_1_users[:,0]
timesteps_3 = rewards_throughput_energy_3_users[:,0]
timesteps_5 = rewards_throughput_energy_5_users[:,0]

rewards_1_users = rewards_throughput_energy_1_users[:,1]
rewards_3_users = rewards_throughput_energy_3_users[:,1]
rewards_5_users = rewards_throughput_energy_5_users[:,1]


range_start_1 = 800
range_finish_1 = 4500

range_start_3 = 400
range_finish_3 = 2600

range_start_5 = 400
range_finish_5 = 1500

timesteps_1_ = timesteps_1[range_start_1:range_finish_1]
timesteps_3_ = timesteps_3[range_start_3:range_finish_3]
timesteps_5_ = timesteps_5[range_start_5:range_finish_5]

rewards_1_users_ = rewards_1_users[range_start_1:range_finish_1]
rewards_3_users_ = rewards_3_users[range_start_3:range_finish_3]
rewards_5_users_ = rewards_5_users[range_start_5:range_finish_5]


#print(timesteps_)

figure, axis = plt.subplots(3,1)
#plt.plot(timesteps_, rewards_1_users_,color = "red")
#plt.plot(timesteps_, rewards_3_users_,color = "green")
#plt.plot(timesteps_, rewards_5_users_,color = "blue")


axis[0].plot(timesteps_1_, rewards_1_users_)
axis[0].set_title('1 User reward')

axis[1].plot(timesteps_3_, rewards_3_users_)
axis[1].set_title('3 Users reward')

axis[2].plot(timesteps_5_, rewards_5_users_)
axis[2].set_title('5 Users reward')

'''
axis[3].plot(timesteps_, fairness_index)
axis[3].set_title('fairness index')
'''


'''
axis[3].scatter(timesteps, offload_actions)
axis[3].set_title('offlaoding actions')

axis[4].scatter(timesteps, power_actions)
axis[4].set_title('power actions')

axis[5].scatter(timesteps, subcarrier_actions)
axis[5].set_title('subcarrier actions')
'''


plt.show()