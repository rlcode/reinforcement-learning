#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import numpy as np
import tensorflow as tf
import pylab
import gym
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
import sys

import matplotlib
from matplotlib import rcParams, pyplot as plt
import argparse
import time
timestr = time.strftime("%d.%m.%Y - %H:%M:%S")

# TODO: We have a problem with the predict functions at actor and critic!

# global variables for threading
episode = 0
scores = []
# We do a for in range, that's why we need 1001 instead of 1000
EPISODES = 10

# Code pulled from https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/4-actor-critic
# A3C agent for Cartpole (Windows OS)


def handleArguments():
    """Handles CLI arguments and saves them globally"""
    # TODO: enable train-mode and test-mode (or else not necessary because we don't need to enforce exploration for stochastic policy)
    parser = argparse.ArgumentParser(
        description="Switch between modes in A2C or loading models from previous games")
    parser.add_argument("--demo_mode", "-d", help="Renders the gym environment", action="store_true")
    parser.add_argument("--load_model", "-l", help="Loads the model of previously gained training data", action="store_true")
    global args
    args = parser.parse_args()

class A3CAgent:
    def __init__(self, state_size, action_size, env_name):

        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # get gym environment name
        self.env_name = env_name

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.hidden1, self.hidden2 = 24, 24
        self.threads = 4

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model or args.load_model:
            self.actor.load_weights("./save_model/a2c_cart_actor.h5")
            self.critic.load_weights("./save_model/a2c_cart_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self):
        agents = [Agent(i, self.actor, self.critic, self.env_name, self.discount_factor,
                        self.action_size, self.state_size) for i in range(self.threads)]

        for agent in agents:
            agent.start()

        time.sleep(5)
        self.actor.save_weights("./save_model/a2c_cart_actor.h5")
        self.critic.save_weights("./save_model/a2c_cart_critic.h5")



# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, actor, critic, env_name, discount_factor, action_size, state_size):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size

    # Thread interactive with environment
    def run(self):
        global episode
        print(threading.current_thread())
        env = gym.make(self.env_name)
        while episode < EPISODES:
            state = env.reset()
            score = 0
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward

                self.memory(state, action, reward)

                state = next_state

                if done:
                    episode += 1
                    print("episode: ", episode, "/ score : ", score)
                    scores.append(score)
                    scores.append(score)
                    self.train_episode(score != 500)
                    break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)
# TODO: We have a problem with the predict functions at actor and critic!
        #values = self.critic.predict(np.array(self.states))
        #values = np.reshape(values, len(values))

        #advantages = discounted_rewards - values

        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        # TODO: We have a problem with the predict functions at actor and critic!
        #policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        return np.random.choice(self.action_size, 1)[0]



if __name__ == "__main__":
    handleArguments()
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # create plotter for windows os
    rcParams.update({'figure.autolayout': True})
    fig, fft_plot = plt.subplots()
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    global_agent = A3CAgent(state_size, action_size, env_name)
    global_agent.train_model()

    # plot episodes on x-axis and the score on y-axis
    fft_plot.set_xlabel("Episodes", fontsize=18)
    fft_plot.set_ylabel("Score", fontsize=18)
    plot = scores[:]
    pylab.plot(range(len(plot)), plot, 'b')
    pylab.show()

    sys.exit()

