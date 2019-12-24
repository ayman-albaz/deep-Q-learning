import keras
import numpy as np

from environment import *

MODEL_NAME = 'models/4x256-1576889586/4x256___-87.33max_-104.55avg_-122.80min__1576889790.model'
model = keras.models.load_model(MODEL_NAME)

done = False
state = env.reset()
prediction = model.predict(np.array(state).reshape(-1, *state.shape))[0]
action = np.argmax(prediction)
while not done:
    env.render()
    new_state, reward, done = env.step(action)[:3]
    prediction = model.predict(np.array(new_state).reshape(-1, *new_state.shape))[0]
    action = np.argmax(prediction)