import gym
from gym import spaces
import numpy as np

class ResourceAllocationEnv(gym.Env):
    def __init__(self, num_users, num_time_slots, num_resource_blocks):
        super(ResourceAllocationEnv, self).__init__()

        self.num_users = num_users
        self.num_time_slots = num_time_slots
        self.num_resource_blocks = num_resource_blocks
        self.num_mini_slots = 2  # Each time slot can be split into 2 mini-slots

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(num_users * self.num_mini_slots * num_resource_blocks)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_users, self.num_mini_slots, num_resource_blocks), dtype=np.int8)

        self.channel_matrix = None

    def reset(self):
        self.channel_matrix = np.zeros((self.num_users, self.num_mini_slots, self.num_resource_blocks), dtype=np.int8)
        return self.channel_matrix

    def step(self, action):
        # Convert action to user_id, mini_slot, and resource_block
        user_id = action // (self.num_mini_slots * self.num_resource_blocks)
        mini_slot = (action % (self.num_mini_slots * self.num_resource_blocks)) // self.num_resource_blocks
        resource_block = action % self.num_resource_blocks

        # Check if the mini-slot is already occupied
        if np.sum(self.channel_matrix[:, mini_slot, resource_block]) == 0:
            self.channel_matrix[user_id, mini_slot, resource_block] = 1

        # Check if every mini-slot is occupied and every user is allocated at least 1 mini-slot
        if np.all(np.sum(self.channel_matrix, axis=1) > 0) and np.all(np.sum(self.channel_matrix, axis=(0, 2)) > 0):
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.channel_matrix, reward, done, {}

    def render(self):
        print(self.channel_matrix)

# Example usage:
num_users = 3
num_time_slots = 5
num_resource_blocks = 6

env = ResourceAllocationEnv(num_users, num_time_slots, num_resource_blocks)

# Reset the environment
state = env.reset()
print("Initial State:")
env.render()

# Perform some actions
for _ in range(num_users * env.num_mini_slots * num_resource_blocks):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.render()
