# -*- coding: utf-8 -*-

import copy
import gym
import gym_ple
# from PIL import Image, ImageOps

import numpy as np
from keras.models import Sequential  # , model_from_json
from keras.layers import Dense, Convolution2D, Flatten, Input
from keras.optimizers import RMSprop

FRAME_WIDTH = 28   # Resized frame width
FRAME_HEIGHT = 28  # Resized build_network(self)
STATE_LENGTH = 4
episodes = 5000


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = []
        self.gamma = 0.9  # decay rate
        self.epsilon = 1  # exploration
        self.epsilon_decay = .99
        self.epsilon_min = 0.2
        self.learning_rate = 0.0001
        print("action space: ", self.env.action_space)
        self._build_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4),
                  border_mode='same', activation='relu', input_shape=(64,64)))
        model.add(Convolution2D(32, 4, 4, subsample=(2, 2),
                  border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(3))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = env.action_space.sample()
            print("explore: ", a)
            return a
        act_values = self.model.predict(state)
        print("predict: ", np.argmax(act_values[0]), "values:", act_values[0])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batchs)
        for i in batchs:
            state, action, reward, next_state = self.memory[i]
            target = reward
            if i != len(self.memory) - 1:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # print("action: ", action, " target: ", target,
            # "target_f: ", target_f)
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('Catcher-v0')
    agent = DQNAgent(env)
    # agent.load("./save/catcher-v0.keras")

    for e in range(episodes):
        state = env.reset()
        # state = np.reshape(state, [1, 4])
        for time_t in range(5000):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state)
            state = copy.deepcopy(next_state)
            if done:
                print("episode: {}/{}, score: {}, memory size: {}, e: {}"
                      .format(e, episodes, time_t,
                              len(agent.memory), agent.epsilon))
                break
        if e % 10 == 0:
            agent.save("./save/catcher-v0.keras")
        if e % 500 == 0:
            agent.save("./save/catcher_backup"+str(e)+"-v0.keras")

        agent.replay(628)
