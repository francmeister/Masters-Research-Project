from gym.envs.registration import register

register(id='NetworkEnv-v0',
         entry_point='Network_Env.envs:NetworkEnv')