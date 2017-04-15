# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

episodes = 1000

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9  # decay rate
        self.epsilon = 1  # exploration
        self.epsilon_decay = .999999
        self.epsilon_min = 0.05
        self.learning_rate = 0.0001
        self._build_model()
        self.action_space = [0, 1, 2, 3, 4]

    def _build_model(self):
        # Neural Net for Deep Q Learning

        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # Dense is the basic form of a neural network layer
        # Input Layer 4 and Hidden Layer with 128 nodes
        model.add(Dense(40, input_dim=52, activation='tanh'))

        # Output Layer with 2 nodes
        model.add(Dense(5, activation='linear'))

        # Create the model based on the information above
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))

        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.choice(self.action_space)

        # Predict the reward value based on the given state
        act_values = self.model.predict(state)

        # Pick the action based on the predicted reward
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)
        for i in batches:
            # Extract informations from i-th index of the memory
            state, action, reward, next_state, done = self.memory[i]

            # if done, make our target reward (-100 penality)
            target = reward

            if not done:
                # predict the future discounted reward
                target = reward + self.gamma * \
                                  np.amax(self.model.predict(next_state)[0])

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
