import gym
import pylab
import random
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D

EPISODES = 5000


class DQNAgent:
    def __init__(self):
        self.render = True

        self.state_size = (84, 84, 4)
        self.action_size = 6

        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 1000000.
        self.epsilon_decay_step = \
            (self.epsilon_start - self.epsilon_end) / self.epsilon_decay

        self.batch_size = 32
        self.train_start = 20000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        self.learning_rate = 0.00025
        self.momentum = 0.95
        self.min_gradient = 0.01

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
        model.add(Dense(self.action_size))
        model.summary()
        model.compile(loss='mse', optimizer=RMSprop(
            lr=self.learning_rate, rho=self.momentum, epsilon=self.min_gradient))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        history = np.float32(history/255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def replay_memory(self, history, action, reward, history1, done):
        self.memory.append((history, action, reward, history1, done))
        if self.epsilon > self.epsilon_end:
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
            history1 = np.float32(history1/255.)
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


def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant')*255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v3')
    agent = DQNAgent()

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        score, start_live = 0, 5
        observe = env.reset()
        next_observe = observe
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe = next_observe
            next_observe, _, _, _ = env.step(1)

        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = history.reshape(1, history.shape[0], history.shape[1], history.shape[2])

        while not done:
            if agent.render:
                env.render()
            observe = next_observe
            action = agent.get_action(history)
            next_observe, reward, done, info = env.step(action)
            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            history1 = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_live > info['ale.lives']:
                dead = True
                start_live = info['ale.lives']

            agent.replay_memory(history, action, reward, history1, done)
            agent.train_replay()

            score += reward

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
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
                pylab.savefig("./save_graph/Breakout_DQN.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

        # 20 에피소드마다 학습 모델을 저장
        if e % 1000 == 0:
            agent.save_model("./save_model/Breakout_DQN.h5")