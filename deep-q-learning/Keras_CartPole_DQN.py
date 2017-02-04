# -*- coding: utf-8 -*-

import gym
from gym.wrappers import Monitor

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd

episodes = 5000


class DQNAgent:
    def __init__(self):
        self.memory = []
        self.gamma = 0.9  # decay rate
        self.epsilon = 0.2  # exploration
        self.epsilon_decay = 0.95
        self.learning_rate = 0.0001
        self._build_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = Sequential()
        model.add(Dense(160, input_dim=4, activation='relu'))
        model.add(Dense(160, activation='relu'))
        model.add(Dense(160, activation='relu'))
        model.add(Dense(2))
        model.compile(loss='mse', optimizer=sgd(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        act_values = self.model.predict(state)
        if np.random.uniform() <= self.epsilon:
            return np.random.choice([0, 1])  # 0 or 1
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batch_size)
        for i in batchs:
            state, action, reward, next_state = self.memory[i]
            target = reward
            if i != len(self.memory) - 1:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            print(state, target_f)
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    agent = DQNAgent()
    env = gym.make('CartPole-v0')
    env = Monitor(env, '/tmp/cartpole-experiment-1', force=True)
    agent.load("./save/cp-v0.keras")

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        for time_t in range(5000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, time_t))
                break
        # if e % 10 == 0:
            # agent.save("./save/cp-v0.keras")
        # if e % 500 == 0:
            # agent.save("./save/cp_backup"+str(e)+"-v0.keras")
        agent.replay(256)
