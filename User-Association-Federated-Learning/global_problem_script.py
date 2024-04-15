import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import gym
import time
import threading
from eMBB_UE import eMBB_UE
from URLLC_UE import URLLC_UE
from numpy import interp
import math

# Initializing User Association DNN
## Defining the neural network class
class UserAssociationDNN(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(UserAssociationDNN, self).__init__()
         self.fc1 = nn.Linear(input_dim, 64)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(64, 32)
         self.fc3 = nn.Linear(32, output_dim)
         self.sigmoid = nn.Sigmoid()

     def forward(self, x):
         x = self.relu(self.fc1(x))
         x = self.relu(self.fc2(x))
         x = self.sigmoid(self.fc3(x))
         return x

# Define some environment varibles
num_embb_users = 4
num_urllc_users = 6
user_count = 1
embb_user_count = 1
urllc_user_count = 1
all_users = []
num_access_points = 3
num_users = num_embb_users+num_urllc_users
num_batches = num_users*5
num_task_arrival_rate = 1

# Create user objects
for x in range(0,num_embb_users):
   embb_user = eMBB_UE(embb_user_count,user_count,100,600)
   all_users.append(embb_user)
   embb_user_count+=1
   user_count+=1

for x in range(0,num_urllc_users):
   embb_user = eMBB_UE(urllc_user_count,user_count,100,600)
   all_users.append(embb_user)
   urllc_user_count+=1
   user_count+=1

# Functions for producing a set of user association schemes based on random exploration
def random_exploration(num_users,num_access_points,num_batches):
   user_association_labels = []
   user_association_labels_ = []
   count = 0
   
   while count < num_batches:
      user_association_labels = np.random.randint(2, size=(num_users, num_access_points))
     
      if np.all(np.sum(user_association_labels,axis=1) == 1):
         user_association_labels_.append(user_association_labels)
         count+=1
        
   user_association_labels_ = np.array(user_association_labels_)
   user_association_labels_for_model_training = user_association_labels_.reshape(num_batches,num_access_points*num_users)
   return user_association_labels_,user_association_labels_for_model_training


def generate_user_association_schemes(users,num_users,num_access_points,num_batches):
   user_association_labels, user_association_labels_for_model_training = random_exploration(num_users,num_access_points,num_batches)

   for user_association_scheme in user_association_labels:
      count = 0
      for user in all_users:
         user.assigned_access_point_label_matrix.append(user_association_scheme[count])

   for user in all_users:
      user.assigned_access_point_label_matrix_to_numpy_array()

   return users, user_association_labels, user_association_labels_for_model_training

   
# Initializing DRL model classes
## Experience Replay Memory
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
  
# Actor Model
class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.sigmoid(self.layer_3(x))
    #x = self.max_action * torch.tanh(self.layer_3(x))
    return x
  
# Critic Models
class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1
  
# TD3 Training class
  
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=0.0000001)
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.0001)
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state).to(device)
    #return self.actor(state).cpu().data.numpy().flatten()
    return self.actor(state).cpu().data.numpy()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      batch_states = np.reshape(batch_states,(batch_states.shape[0]*batch_states.shape[1],batch_states.shape[2]))
      batch_next_states = np.reshape(batch_next_states,(batch_next_states.shape[0]*batch_next_states.shape[1],batch_next_states.shape[2]))
      batch_actions = np.reshape(batch_actions,(batch_actions.shape[0]*batch_actions.shape[1],batch_actions.shape[2]))

      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, torch.Tensor(next_action).to(device))
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

# Function to evaluate the model
def evaluate_policy(policy, env, eval_episodes=1):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(obs)
      action = env.reshape_action_space_from_model_to_dict(action)
      obs, reward, done, _ = env.step(action)
      done = done[len(done)-1]
      avg_reward += sum(reward)#interp(sum(reward),[720000000,863000000],[0,1000])
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

# Function to assign users to their allocated access points
def find_access_point_users(users,access_point_id,user_association_epoch_number):
   access_point_users = []
   for user in users:
      if user.assigned_access_point_label_matrix_integers[user_association_epoch_number] == access_point_id:
         access_point_users.append(user)
   return access_point_users

# Function to perform training on the DRL model(s)
def training(policy,replay_buffer,env, access_point_id):
    results_folder_name = "./results/access_point_%d" % (access_point_id)
    model_save_folder_name = "./pytorch_models/access_point_%d" % (access_point_id)
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    if save_models and not os.path.exists(model_save_folder_name):
        os.makedirs(model_save_folder_name)

    env_name = "Access Point " + str(access_point_id) # Name of a environment (set it to any Continous environment you want)
    seed = 0 # Random seed number
    start_timesteps = 10000 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5000 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 500000 # Total number of iterations/timesteps
    save_models = True # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

    file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
    file_name_0 = "Evaluations"
    file_name_1 = "timestep_rewards_energy_throughput"
    file_name_2 = "offloading_actions"
    file_name_3 = "power_actions"
    file_name_4 = "subcarrier_actions"
    file_name_5 = "allocated_RBs"
    file_name_6 = "fairnes_index"

    file_name_7 = "energy_efficiency_rewards"
    file_name_8 = "battery_energy_rewards"
    file_name_9 = "throughput_rewards"
    file_name_10 = "delay_rewards"
    file_name_11 = "sum_allocations_per_RB_matrix"
    file_name_12 = "RB_allocation_matrix"
    file_name_13 = "energy_rewards"
    file_name_14 = "delays"
    file_name_15 = "tasks_dropped"
    file_name_16 = "resource_allocation_matrix"
    file_name_17 = "resource_allocation_constraint_violation_count"

    evaluations = [evaluate_policy(policy, env)]
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    timestep_rewards = []
    timestep_rewards_energy_throughput = []
    offload_actions = []
    power_actions = []
    subcarrier_actions = []
    allocated_RBs = []
    fairness_index = []
    energy_efficiency_rewards = []
    battery_energy_rewards = []
    energy_rewards = []
    throughput_rewards = []
    delay_rewards = []
    sum_allocations_per_RB_matrix = []
    change_action = 0
    RB_allocation_matrix = []
    delays = []
    tasks_dropped = []
    resource_allocation_matrix = []
    resource_allocation_constraint_violation_count = []
    # We start the main loop over 500,000 timesteps
    while total_timesteps < max_timesteps:
    
    # If the episode is done
        if done:
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                timestep_rewards.append([total_timesteps, episode_reward])
                timestep_rewards_energy_throughput.append([total_timesteps,episode_reward,env.total_energy,env.total_rate])
                offload_actions.append(env.offload_decisions)
                power_actions.append(env.powers)
                subcarrier_actions.append(env.subcarriers)
                allocated_RBs.append(env.Communication_Channel_1.allocated_RBs)
                fairness_index.append(env.SBS1.fairness_index)

                energy_efficiency_rewards.append(env.SBS1.energy_efficiency_rewards)
                battery_energy_rewards.append(env.SBS1.battery_energy_rewards)
                throughput_rewards.append(env.SBS1.throughput_rewards)
                delay_rewards.append(env.SBS1.delay_rewards)
                sum_allocations_per_RB_matrix.append(env.sum_allocations_per_RB_matrix)
                RB_allocation_matrix.append(env.RB_allocation_matrix)
                energy_rewards.append(env.SBS1.energy_rewards)
                delays.append(env.SBS1.delays)
                tasks_dropped.append(env.SBS1.tasks_dropped)
                resource_allocation_matrix.append(env.resource_block_allocation_matrix)
                resource_allocation_constraint_violation_count.append(env.resource_allocation_constraint_violation)
                
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(evaluate_policy(policy))
                policy.save(file_name, directory="./pytorch_models/access_point_%d" % (access_point_id))
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_0), evaluations)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_1), timestep_rewards_energy_throughput)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_2), offload_actions)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_3), power_actions)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_4), subcarrier_actions)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_5), allocated_RBs)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_6), fairness_index)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_7), energy_efficiency_rewards)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_8), battery_energy_rewards)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_9), throughput_rewards)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_10), delay_rewards)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_11), sum_allocations_per_RB_matrix)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_12), RB_allocation_matrix)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_13), energy_rewards)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_14), delays)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_15), tasks_dropped)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_16), resource_allocation_matrix)
                np.save("./results/access_point_%d/%s" % (access_point_id,file_name_17), resource_allocation_constraint_violation_count)

            # When the training step is done, we reset the state of the environment
            obs = env.reset()
            
            # Set the Done to False
            done = False
            
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
            if change_action == 0:
                action = env.action_space.sample()
                action = env.enforce_constraint(action)
                change_action = 1
            else:
                action = env.action_space.sample()
                action = env.enforce_constraint(action)
                change_action = 0

        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape)).clip(env.action_space_low, env.action_space_high)

            #print(action)
            action = env.reshape_action_space_from_model_to_dict(action)
            #action = env.enforce_constraint(action)
            #print(action)
            
        
        #print("Action in training")
        #print(action)
        #print(' ')
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, dones, _ = env.step(action)
        done = dones[len(dones) - 1]
        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == env.STEP_LIMIT else float(done)
        
        # We increase the total reward
        episode_reward += sum(reward)
        #print('episode reward')
        #print(episode_reward)
        #episode_reward = interp(episode_reward,[720000000,863000000],[0,1000])
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        action = env.reshape_action_space_for_model(action)
        replay_buffer.add((obs, new_obs, action, reward, dones[:len(dones)-1] + [done_bool]))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

        # We add the last policy evaluation to our list of evaluations and we save our model
        evaluations.append(evaluate_policy(policy))
        if save_models: policy.save("%s" % (file_name), directory="./pytorch_models/access_point_%d" % (access_point_id))
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_0), evaluations)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_1), timestep_rewards_energy_throughput)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_2), offload_actions)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_3), power_actions)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_4), subcarrier_actions)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_5), allocated_RBs)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_6), fairness_index)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_7), energy_efficiency_rewards)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_8), battery_energy_rewards)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_9), throughput_rewards)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_10), delay_rewards)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_11), sum_allocations_per_RB_matrix)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_12), RB_allocation_matrix)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_13), energy_rewards)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_14), delays)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_15), tasks_dropped)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_16), resource_allocation_matrix)
        np.save("./results/access_point_%d/%s" % (access_point_id,file_name_17), resource_allocation_constraint_violation_count)

# Calculate total reward across all access points at the end of training
def calculate_total_system_reward(num_access_point):
   evaluations = []
   access_point_id = 0
   file_name_0 = "Evaluations"
   for x in range(1, num_access_point+1):
      access_point_id = x
      evaluation = np.load("./results/access_point_%d/%s" % (access_point_id,file_name_0), evaluations)
      evaluations.append(evaluation)

   total_system_reward = 0

   for evaluation in evaluations:
      total_system_reward+= evaluation[len(evaluation)-1]
   return total_system_reward

# min_channel_gain = math.pow(10,-5)
# max_channel_gain = 10
# print(large_channel_gains)
# for r in range(0,total_row_count):
#     for c in range(0,total_column_count):
#         large_channel_gains[r][c] = interp(large_channel_gains[r][c],[max_channel_gain,min_channel_gain],[0,1])

# Prepare training data for user association DNN
def prepare_data_for_training_DNN(all_users, num_users,num_access_points):
   large_channel_gains = []
   task_arrival_rates = []

   for user in all_users:
      large_channel_gains.append(user.initial_large_scale_gain_all_access_points(num_access_points))
      task_arrival_rates.append(user.initial_arrival_rates())

   large_channel_gains = np.array(large_channel_gains)
   total_row_count = large_channel_gains.shape[0]
   total_column_count = large_channel_gains.shape[1]
   total_num_features = total_row_count*total_column_count
   max_channel_gain = large_channel_gains.max()

   task_arrival_rates = np.array(task_arrival_rates)
   user_features = np.column_stack((large_channel_gains,task_arrival_rates))
   total_row_count = user_features.shape[0]
   total_column_count = user_features.shape[1]
   total_num_features = total_row_count*total_column_count
   user_features = np.transpose(user_features)
   user_features = user_features.reshape(1,total_num_features)
   user_features = user_features.squeeze()

   return user_features

# Prepare variables necessary to train the DNN
user_features = prepare_data_for_training_DNN(all_users, num_users,num_access_points)
all_users, user_association_schemes, user_association_schemes_for_model_training = generate_user_association_schemes(all_users,num_users,num_access_points,num_batches)

num_features = len(user_features)
num_outputs = num_access_points*num_users

#  # Instantiate the model
User_Association_DNN = UserAssociationDNN(input_dim=num_features, output_dim=num_outputs)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(User_Association_DNN.parameters(), lr=0.001)

num_training_epochs = 10
training_memory = []

# Start training the DNN
for epoch in range(0,num_training_epochs):
   if epoch != 0:
      user_features = prepare_data_for_training_DNN(all_users, num_users,num_access_points)
      all_users, user_association_schemes, user_association_schemes_for_model_training = generate_user_association_schemes(all_users,num_users,num_access_points,num_batches)

   user_association_epoch_number = 0
   total_system_reward = 0
   total_system_rewards = []
   for i in range(0,user_association_schemes_for_model_training):
      access_points_envs = []
      access_point_users = []
      user_association_epoch_number = i
      policies = []
      replay_buffers = []
      threadlist = []

      state_dim = 0
      action_dim = 0
      max_action = 0
      seed = 0

      for x in range(1,num_access_points+1):
         access_point_users = find_access_point_users(all_users,x,user_association_epoch_number)
         env = gym.make(x,access_point_users)
         access_points_envs.append(env)

        
      for x in range(0,num_access_points):
         torch.manual_seed(seed)
         np.random.seed(seed)
         state_dim = access_points_envs[x].observation_space.shape[1]
         action_dim = access_points_envs[x].action_space_dim_1
         max_action = float(access_points_envs[x].box_action_space.high[0][1]) # to change this soon

         policy = TD3(state_dim, action_dim, max_action)
         replay_buffer = ReplayBuffer()

         policies.append(policy)
         replay_buffers.append(replay_buffer)

      for x in range(0,num_access_points):
         t = threading.Thread(target=training,args=(policies[x],replay_buffers[x],access_points_envs[x],x)) 
         threadlist.append(t)
         t.start()

      for tr in threadlist:
         tr.join()

      total_system_reward = calculate_total_system_reward(num_access_points)
      total_system_rewards.append(total_system_reward)

   max_index = np.array(total_system_rewards).argmax()
   y_true = user_association_schemes_for_model_training[max_index]
   training_memory.append((user_features,y_true))

   user_features_for_training = []
   y_true_values_for_training = []

   for training_sample in training_memory:
      user_features_for_training.append(training_sample[0])
      y_true_values_for_training.append(training_sample[1])

   user_features_for_training = np.array(user_features_for_training)
   y_true_values_for_training = np.array(y_true_values_for_training)

   user_features_for_training_tensor = torch.from_numpy(user_features_for_training)
   y_true_values_for_training_tensor = torch.from_numpy(y_true_values_for_training)

   y_pred_tensor = User_Association_DNN(user_features_for_training_tensor)

   loss = criterion(y_pred_tensor, y_true_values_for_training_tensor)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   

      

      








