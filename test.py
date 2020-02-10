import gym
import numpy as np
import random

for i in range(100):
    print(random.randint(0,7))
# observation_space = gym.spaces.Box(low=0, high=10000, shape=(19,))  # Box(84,84,4)
# action_space = gym.spaces.Box(low=0, high=10000, shape=(7,))  # Discrete(4)
#
# rt_observation_space = gym.spaces.Tuple((observation_space, action_space))
# assert isinstance(rt_observation_space, gym.spaces.Tuple)
# print(rt_observation_space, type(rt_observation_space))
# print(observation_space.shape, action_space.shape, type(observation_space.shape))
# input_dim = sum(s.shape[0] for s in rt_observation_space)

