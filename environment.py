import gym

env = gym.make('MountainCar-v0')
env.seed(7)

def custom_reward(observation):
    return -(((abs(env.observation_space.high[1]) - abs(observation[1])) / abs(env.observation_space.high[1])) ** (1/2))

env.compute_reward = custom_reward