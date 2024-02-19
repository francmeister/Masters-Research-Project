import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.layer_1 = nn.Linear(state_dim, 400)
		self.layer_2 = nn.Linear(400, 300)
		self.layer_3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		x = F.relu(self.layer_1(state))
		x = F.relu(self.layer_2(x))
		return self.max_action * torch.sigmoid(self.layer_3(x))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.layer_1 = nn.Linear(state_dim + action_dim, 400)
		self.layer_2 = nn.Linear(400, 300)
		self.layer_3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.layer_1(torch.cat([state, action], 1)))
		q = F.relu(self.layer_2(q))
		return self.layer_3(q)


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau

	def select_action(self, state):
		#state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		state = torch.Tensor(state).to(device)
		return self.actor(state).cpu().data.numpy()
		#return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=100):
		# Sample replay buffer 
	    batch_states, batch_next_states, batch_actions,batch_rewards, batch_dones = replay_buffer.sample(batch_size)
        
        state = torch.Tensor(batch_states).to(device)
        next_state = torch.Tensor(batch_next_states).to(device)
        action = torch.Tensor(batch_actions).to(device)
        reward = torch.Tensor(batch_rewards).to(device)
        done = torch.Tensor(batch_dones).to(device)
      

		# Compute the target Q value
	    next_action = self.actor_target(next_state)	
		target_Q = self.critic_target(next_state, next_action)
		target_Q = reward + ((1-done) * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		