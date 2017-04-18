import gym
import pylab
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from collections import deque
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.convolutional import Conv2D

EPISODES = 5000


class DQNAgent:
    def __init__(self):
        self.render = True

        self.state_size = (80, 80, 4)
        self.action_size = 6

        self.epsilon = 1
        self.epsilon_start = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 50000.
        self.epsilon_decay_step = \
            (self.epsilon_start - self.epsilon_end) / self.epsilon_decay

        self.batch_size = 32
        self.train_start = 3200
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=50000)

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
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def replay_memory(self, history, action, reward, history1, done):
        self.memory.append((history, action, reward, history1, done))
        if len(self.memory) > 3200:
            self.epsilon -= self.epsilon_decay_step

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            history, action, reward, history1, done = mini_batch[i]
            history = np.float32(history/255.)
            target = self.model.predict(history)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * np.amax(self.target_model.predict(history1)[0])
            update_target[i] = target
            update_input[i] = history

        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


def pre_processing(observe):
    observe = rgb2gray(observe)
    observe = resize(observe, (80, 80), mode='constant')
    observe = rescale_intensity(observe, out_range=(0, 255))
    return observe


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v3')
    agent = DQNAgent()

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        score, start_live = 0, 5
        observe = env.reset()
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = history.reshape(1, history.shape[0], history.shape[1], history.shape[2])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(history)
            next_observe, reward, done, info = env.step(action)
            next_state = pre_processing(next_observe)
            next_state = np.reshape([next_state], (1, 80, 80, 1))
            history1 = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_live > info['ale.lives']:
                dead = True
                start_live = info['ale.lives']
                reward = -1

            agent.replay_memory(history, action, reward, history1, done)
            agent.train_replay()

            score += reward

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 80, 80, 4))
                dead = False
            else:
                history = history1

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            if done:
                env.reset()
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Breakout_DQN1.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)