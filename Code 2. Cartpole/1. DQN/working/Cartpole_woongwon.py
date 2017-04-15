import gym
import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = "True"
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.9
        self.learning_rate = 0.01

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(16, activation='tanh', init='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def get_action(self):
        pass

    def replay_memory(self):
        pass

    def train_replay(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print("state:", state, "  action:", action, "  reward:", reward)
        if done:
            env.reset()
