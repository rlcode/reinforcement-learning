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
        self.discount_factor = 0.99
        self.learning_rate = 0.00025

    def build_model(self):
        model = Sequential()


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
        print("action: ", action, "  reward: ", reward)
        if done:
            env.reset()
