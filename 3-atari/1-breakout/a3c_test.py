import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K

global episode
episode = 0
EPISODES = 8000000
env_name = "BreakoutDeterministic-v4"


class Agent:
    def __init__(self, action_size):
        self.render = False
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        # build
        self.actor, self.critic = self.build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.summary()
        critic.summary()

        return actor, critic


    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]

        policy = policy - np.finfo(np.float32).epsneg

        action_index = np.argmax(policy)
        return action_index, policy

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    agent = Agent(action_size=3)

    env = gym.make(env_name)
    agent.load_model("save_model/breakout_a3c_5")

    step = 0

    while episode < EPISODES:
        done = False
        dead = False
        # 1 episode = 5 lives
        score, start_life = 0, 5
        observe = env.reset()
        next_observe = observe

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, 20)):
            observe = next_observe
            next_observe, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame. So just copy initial states to make history
        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:	
            env.render()
            step += 1
            observe = next_observe
            # get action for the current history and go one step in environment
            action, policy = agent.get_action(history)
            
            if action == 1:
                fake_action = 2
            elif action == 2:
                fake_action = 3
            else: fake_action = 1

            if dead:
                fake_action = 1
                dead = False

            next_observe, reward, done, info = env.step(fake_action)
            # pre-process the observation --> history
            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the ball is fall, then the agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                reward = -1
                start_life = info['ale.lives']

            score += reward

            # if agent is dead, then reset the history
            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                episode += 1
                print("episode:", episode, "  score:", score, "  step:", step)
                step = 0
