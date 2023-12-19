import gym
from gym import spaces
import numpy as np

class ResourceAllocationEnv(gym.Env):
    def __init__(self, num_users, num_resource_blocks):
        super(ResourceAllocationEnv, self).__init__()
        self.num_users = num_users
        self.num_resource_blocks = num_resource_blocks

        # Define the action space: 2D multi-binary array representing user vs resource allocation
        self.action_space = spaces.MultiBinary(self.num_users * self.num_resource_blocks)

        # Define the observation space (same as the action space for simplicity)
        self.observation_space = spaces.MultiBinary(self.num_users * self.num_resource_blocks)

        # Initialize user resource block allocations
        self.user_allocations = np.zeros((self.num_users, self.num_resource_blocks), dtype=np.int32)

    def step(self, action):
        # Reshape the action to the 2D allocation shape
        action_matrix = action.reshape(self.num_users, self.num_resource_blocks)

        # Ensure uniqueness within columns (resource block allocation)
        if not np.all(np.sum(action_matrix, axis=0) <= 1):
            return self.user_allocations.flatten(), 0, True, {}  # Return invalid action if any block is allocated to multiple users

        # Update resource block allocations for users based on the action
        self.user_allocations = action_matrix

        # Calculate reward (for illustration purposes, you may define a custom reward function)
        reward = np.sum(action)  # Example: Reward based on the sum of allocated blocks

        # Define a boolean indicating if the episode has ended
        done = False

        # Placeholder for observation (in this simple example, it's the same as the action)
        observation = self.user_allocations.flatten()

        return observation, reward, done, {}

    def reset(self):
        # Reset the environment
        self.user_allocations = np.zeros((self.num_users, self.num_resource_blocks), dtype=np.int32)
        return self.user_allocations.flatten()

    def render(self, mode='human'):
        # Rendering the environment state (for example purposes)
        print(f"User allocations:\n{self.user_allocations}")

# Example usage
if __name__ == "__main__":
    num_users = 3
    num_resource_blocks = 6

    env = ResourceAllocationEnv(num_users, num_resource_blocks)
    obs = env.reset()

    for _ in range(3):  # Three steps for illustration
        action = env.action_space.sample()  # Sample random action
        obs, reward, done, _ = env.step(action)
        while done:  # If an invalid action is returned, resample until a valid action is obtained
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
        env.render()
