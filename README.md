# deep-Q-learning
Deep reinforcement Q-learning example with TensorFlow and MountainCar environment


## Variables
All these variables are modifiable. Some interact with each other, so becareful when altering them.

#VIS variables
* RENDER = True
* RENDER_EVERY = 10

#NN variables
* EPISODES = 100
  * The # of episodes the DQN-agent will run for. Not to be confused with TensorFlows 'epochs'.
* MODEL_NAME = '4x256'
  * Every file output will have this as a prefix.
* MIN_REWARD = -200
  * If the model fails to produce a reward greater than MIN_REWARD when it wants to save the model, it will not save the model.
* MINIBATCH_SIZE = 64  
  * The # of events from the replay memory to be used for training in every instance. DQNs work through learning with minibatches.
* MIN_REPLAY_MEMORY_SIZE = 1000
  * The # Learning will only occur once 1000 events have filled up the replay memory. This is to avoid training on an empty memory.
* REPLAY_MEMORY_SIZE = 5000
  * The # Number of events to store in the memory at all times. Replays are made using deque from the default collections library.
* STATS_EVERY = 10
  * Min, max, and mean are calculated and sent to the tensorboard every 10 episodes.
* UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
  * Updates the target every 5 episodes. DQNs have 2 tensorflow models, a current and a target model.

#LEARNING variables
* DISCOUNT = 0.99
* EPSILON = 0.1
* EPSILON_MIN = 0
* EPSILON_DECAY = 0.99975
* LEARNING_RATE = 0.001

## Modifications
I changed the default mountain-car environment around a bit. The default reward mechanism was to give a '-1' reward for every instance that the car doesn't reach the target goal. This reward mechanism is exactly what the 'hypothetical goal' of the environment is. To reach the goal as fast as possible. However, I did not use it as it took too much time to train to initially reach the goal. In other words, the agent would not really train (get any positive or less negative feedback), until the car reaches its goal once.

Therefore the reward I set does the following: ```-(((abs(env.observation_space.high[1]) - abs(observation[1])) / abs(env.observation_space.high[1])) ** (1/2))```.
This equation just means that the car is rewarded (or punished less) whenever its speed is closer to the theoritical maximum speed. Ideally, you'd want to use a combination of both reward systems, however this equation is the fastest way to reach the goal, since we are getting positive feedback (less negative feedback) from the first training instance. 

The equation uses the square root of a fraction of a number between 0 and 1. This means that really fast speeds are rewarded very heavily, and really slow speeds are punished very heavily.

## Testing
You can see how well the models ran by running the test.py file, I added some of the models I made so you can test for yourself.

## Tensorboard
Using the command ```tensorboard --logdir logs``` you are able to view the modified tensorboard.

## Acknowledgement
Shout out to sentdex from youtube. The bases for this DQN and modified tensorboard came from his tutorial series.
