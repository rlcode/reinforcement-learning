# -*- coding: utf-8 -*-

import copy
from collections import deque
import gym

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

episodes = 50001

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9  # decay rate
        self.epsilon = 0.7  # exploration
        self.epsilon_decay = .99
        self.epsilon_min = 0.05
        self.learning_rate = 0.0001
        self._build_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(2, activation='linear', init='he_uniform'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
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
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQNAgent(env)
    agent.load("./save/cartpole-starter.h5")

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        for time_t in range(5000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            reward = -1 if done else reward
            agent.remember(state, action, reward, next_state)
            state = copy.deepcopy(next_state)
            if done:
                print("episode: {}/{}, score: {}, memory size: {}, e: {}"
                      .format(e, episodes, time_t,
                              len(agent.memory), agent.epsilon))
                break
        if e % 10 == 0:
            agent.save("./save/cartpole-v0.h5")
        if e % 10000 == 0:
            agent.save("./save/cartpole_backup"+str(e)+"-v0.h5")
        agent.replay(32)
