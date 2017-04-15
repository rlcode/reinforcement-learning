import gym
import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D


class DQNAgent:
    def __init__(self):
        self.render = "True"
        width, height, length = 84, 84, 4
        self.state_size = width * height * length
        self.action_size = 3  # Left, Right, Stay
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


def pre_processing():
    pass

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v3')
    state = env.reset()
    agent = DQNAgent()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print("action: ", action, "  reward: ", reward)
        if done:
            env.reset()
