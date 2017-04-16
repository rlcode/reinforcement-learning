import gym
import pylab
import random
import numpy as np
from skimage.transform import resize
from collections import deque
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.convolutional import Conv2D

EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = "True"

        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 1000000
        self.epsilon_decay_step = \
            (self.epsilon_start - self.epsilon_end) / self.epsilon_decay

        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.memory = deque(maxlen=1000000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), input_shape=self.state_size, activation='relu', strides=(4, 4),
                         kernel_initializer='glorot_uniform'))
        model.add(Conv2D(64, (4, 4), activation='relu', strides=(2, 2),
                         kernel_initializer='glorot_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                         kernel_initializer='glorot_uniform'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.00025, beta_1=0.95, beta_2=0.95,
                                                 epsilon=0.01, clipnorm=1.))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        if np.random.rand() <= 0:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(np.reshape(np.float32(history[:, :, 0:4] / 255.), [1, 84, 84, 4]))
            print(q_value)
            return np.argmax(q_value[0])

    def replay_memory(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        # print(len(self.memory))

    def train_replay(self):
        #if len(self.memory) < self.train_start:
        #    return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = []
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            history, action, reward, done = mini_batch[i]
            history = np.float32(history/255.)
            target = self.model.predict(np.reshape(history[:, :, 0:4], [1, 84, 84, 4]))[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * np.amax(self.target_model.predict
                                        (np.reshape(history[:, :, 1:5], [1, 84, 84, 4]))[0])
            update_target[i] = target
            update_input.append(np.reshape(history[:, :, 0:4], [1, 84, 84, 4]))

        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


def pre_processing(observe):
    observe = np.uint8(np.dot(observe[:, :, :3], [0.299, 0.587, 0.114]))
    observe = resize(observe, (110, 84), mode='reflect')
    observe = observe[17:101, :]
    return observe


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v3')

    state_size = (84, 84, 4)
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes, global_step = [], [], 0
    history = np.zeros([84, 84, 5], dtype=np.uint8)

    for e in range(EPISODES):
        done = False
        score, start_live = 0, 5
        observe = env.reset()
        state = pre_processing(observe)
        for i in range(4):
            history[:, :, i] = state

        while not done:
            if agent.render == "True":
                env.render()
            action = agent.get_action(history)
            next_observe, reward, done, info = env.step(action)
            next_state = pre_processing(next_observe)
            if start_live > info['ale.lives']:
                done = True
                start_live = info['ale.lives']
                reward = -1

            history[:, :, 4] = next_state

            agent.replay_memory(history[:, :, 0:5], action, reward, done)
            agent.train_replay()

            history[:, :, 0:4] = history[:, :, 1:5]
            score += reward

            if global_step % agent.update_target_rate == 0:
               agent.update_target_model()

            if done:
                env.reset()
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)
