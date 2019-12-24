import numpy as np
import os
import tensorflow as tf
from time import time
from tqdm import tqdm

from DQNAgent import DQNAgent
from environment import env
from variables import *


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
tf.compat.v1.keras.backend.set_session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

np.random.seed(7)
START_TIME = int(time())

if not os.path.exists(f'models/{MODEL_NAME}-{START_TIME}'):
    os.makedirs(f'models/{MODEL_NAME}-{START_TIME}')

agent = DQNAgent()

episodes_reward = []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    
    # Reset environment and get initial state
    current_state = env.reset()
    
    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > EPSILON:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done = env.step(action)[:3]
        reward = env.compute_reward(new_state)
        
        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if RENDER and episode % RENDER_EVERY == 0:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
        
    # Append episode reward to a list and log stats (every given number of episodes)
    episodes_reward.append(episode_reward)
    if episode == 1 or episode % STATS_EVERY == 0:
        average_reward = sum(episodes_reward[-STATS_EVERY:])/len(episodes_reward[-STATS_EVERY:])
        min_reward = min(episodes_reward[-STATS_EVERY:])
        max_reward = max(episodes_reward[-STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=EPSILON)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}-{START_TIME}/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time())}.model')

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(EPSILON_MIN, EPSILON)
