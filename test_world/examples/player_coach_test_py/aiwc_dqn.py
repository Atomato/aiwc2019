import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

# DQN Agent for the soccer robot refering to atari breakout
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # load model if True
        self.load_model = True
        load_add = './save_model/coach_dlck_dqn1032000.h5'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.history_size = 1
        # store the history and the action pair
        self.action = np.int64(0)
        self.history = np.zeros([1, self.state_size, self.history_size])

        # build model
        self.model = self.build_model()

        if self.load_model:
            self.model.load_weights(load_add)
            print('load model form %s' % load_add)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size*self.history_size, 
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        q_value = self.model.predict(history)
        return np.argmax(q_value[0])