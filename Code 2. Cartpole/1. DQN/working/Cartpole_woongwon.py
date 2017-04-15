import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense

EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = "True"

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.9
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05

        self.model = self.build_model()
        self.memory = deque(maxlen=500000)

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(16, activation='tanh', kernel_initializer='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_replay(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(EPISODES):
        done = False
        count = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()
            count += 1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_memory(state, action, reward, next_state, done)
            # print("episode:", e, "  state:", state, "  action:", action, "  reward:", reward)
            if done:
                env.reset()
                print("episode:", e, "  score:", count)