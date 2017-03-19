# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard

episodes = 1000



class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=400000)
        self.gamma = 0.95  # decay rate
        self.epsilon = 1.0  # exploration
        self.epsilon_decay = .9985
        self.epsilon_min = 0.05
        self.learning_rate = 0.00001
        self.model = self._build_model()
        self.action_space = [0, 1, 2, 3, 4]
        #self.tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

    def _build_model(self):
        # Neural Net for Deep Q Learning
        # Sequential() creates the foundation of the layers.
        model = Sequential()
        # Dense is the basic form of a neural network layer
        # Input Layer 4 and Hidden Layer with 128 nodes
        model.add(Dense(150, input_dim=28, activation='tanh'))
        model.add(Dense(150, activation='tanh'))
        # Output Layer with 2 nodes
        model.add(Dense(5, activation='linear'))

        # Create the model based on the information above
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.choice(self.action_space)

        # Predict the reward value based on the given state
        act_values = self.model.predict(state)

        # Pick the action based on the predicted reward
        #print (act_values[0])
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < 240000:
            return
        batches = np.random.choice(len(self.memory), batch_size)
        states, targets = [],[]
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
            states.append(state[0])
            targets.append(target_f[0])
        # Train the Neural Net with the state and target_f
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, nb_epoch=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
