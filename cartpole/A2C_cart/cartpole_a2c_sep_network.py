#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code is made for windows os and works on a virtual environment which has the tensorflow API installed

import sys
import gym
import tensorflow as tf
import matplotlib
from matplotlib import rcParams, pyplot as plt
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

import argparse
from datetime import datetime
import time
timestr = time.strftime("%d.%m.%Y - %H:%M:%S")

# Are these really necessary???
tf.compat.v1.get_default_session()
#tf.compat.v1.assign_add()
#tf.compat.v1.variables_initializer()
tf.compat.v1.global_variables()
tf.compat.v1.Session()
tf.compat.v1.ConfigProto()
#tf.compat.v1.train.Optimizer()

# We do a for in range, that's why we need 1001 instead of 1000
EPISODES = 10

# Code pulled from https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole/4-actor-critic
# A2C(Advantage Actor-Critic) agent for the Cartpole

def handleArguments():
    """Handles CLI arguments and saves them globally"""
    # TODO: enable train-mode and test-mode (or else not necessary because we don't need to enforce exploration for stochastic policy)
    parser = argparse.ArgumentParser(
        description="Switch between modes in A2C or loading models from previous games")
    parser.add_argument("--demo_mode", "-d", help="Renders the gym environment", action="store_true")
    parser.add_argument("--load_model", "-l", help="Loads the model of previously gained training data", action="store_true")
    global args
    args = parser.parse_args()


class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.actor, self.critic = self.build_model()

        if self.load_model or args.load_model:
            self.actor.load_weights("./save_model/a2c_cart_actor.h5")
            self.critic.load_weights("./save_model/a2c_cart_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))

        actor_hidden1 = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)
        critic_hidden1 = Dense(self.hidden1, input_dim=self.state_size, activation='relu',
                              kernel_initializer='glorot_uniform')(state)

        actor_hidden2 = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(actor_hidden1)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden2)

        critic_hidden2 = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(critic_hidden1)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_hidden2)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return actor, critic


    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    starttime = datetime.now()
    handleArguments()
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    # create plotter for windows os
    rcParams.update({'figure.autolayout': True})
    fig, fft_plot = plt.subplots()
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if args.demo_mode:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)

                # plot e on x-axis and the score on y-axis
                fft_plot.set_xlabel("Episode", fontsize=18)
                fft_plot.set_ylabel("Score", fontsize=18)

                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episodes is bigger than 490
                # stop training
                #if np.mean(scores[-min(10, len(scores)):]) > 490:
                #    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.actor.save_weights("./save_model/a2c_cart_actor.h5")
            agent.critic.save_weights("./save_model/a2c_cart_critic.h5")

    # plot the results before exit
    plt.plot(episodes, scores, color='blue')
    plt.show()
    print ()
    endtime = datetime.now()

    print ("Number of Episodes: ", EPISODES, " | Finished within: ", endtime - starttime)
    time.sleep(5)
    sys.exit()
