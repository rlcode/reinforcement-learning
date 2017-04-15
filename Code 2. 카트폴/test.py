import random
from keras.models import Sequential
from keras.layers.core import Dense
from collections import deque

import os
import sys
from time import sleep

import numpy as np
from blessings import Terminal

class Game():
    def __init__(self, shape=(10,10)):
        self.shape = shape
        self.height, self.width = shape
        self.last_row = self.height - 1
        self.paddle_padding = 1
        self.n_actions = 3 # left, stay, right
        self.term = Terminal()
        self.reset()

    def reset(self):
        # reset grid
        self.grid = np.zeros(self.shape)

        # can only move left or right (or stay)
        # so position is only its horizontal position (col)
        self.pos = np.random.randint(self.paddle_padding, self.width - 1 - self.paddle_padding)
        self.set_paddle(1)

        # item to catch
        self.target = (0, np.random.randint(self.width - 1))
        self.set_position(self.target, 1)

    def move(self, action):
        # clear previous paddle position
        self.set_paddle(0)

        # action is either -1, 0, 1,
        # but comes in as 0, 1, 2, so subtract 1
        action -= 1
        self.pos = min(max(self.pos + action, self.paddle_padding), self.width - 1 - self.paddle_padding)

        # set new paddle position
        self.set_paddle(1)

    def set_paddle(self, val):
        for i in range(1 + self.paddle_padding*2):
            pos = self.pos - self.paddle_padding + i
            self.set_position((self.last_row, pos), val)

    @property
    def state(self):
        return self.grid.reshape((1,-1)).copy()

    def set_position(self, pos, val):
        r, c = pos
        self.grid[r,c] = val

    def update(self):
        r, c = self.target

        self.set_position(self.target, 0)
        self.set_paddle(1) # in case the target is on the paddle
        self.target = (r+1, c)
        self.set_position(self.target, 1)

        # off the map, it's gone
        if r + 1 == self.last_row:
            # reward of 1 if collided with paddle, else -1
            if abs(c - self.pos) <= self.paddle_padding:
                return 1
            else:
                return -1

        return 0

    def render(self):
        print(self.term.clear())
        for r, row in enumerate(self.grid):
            for c, on in enumerate(row):
                if on:
                    color = 235
                else:
                    color = 229

                print(self.term.move(r, c) + self.term.on_color(color) + ' ' + self.term.normal)

        # move cursor to end
        print(self.term.move(self.height, 0))


class Agent():
    def __init__(self, env, explore=0.1, discount=0.9, hidden_size=100, memory_limit=5000):
        self.env = env
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(env.height * env.width,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(env.n_actions))
        model.compile(loss='mse', optimizer='sgd')
        self.Q = model

        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)

        self.explore = explore
        self.discount = discount

    def choose_action(self):
        if np.random.rand() <= self.explore:
            return np.random.randint(0, self.env.n_actions)
        state = self.env.state
        q = self.Q.predict(state)
        return np.argmax(q[0])

    def remember(self, state, action, next_state, reward):
        # the deque object will automatically keep a fixed length
        self.memory.append((state, action, next_state, reward))

    def _prep_batch(self, batch_size):
        if batch_size > self.memory.maxlen:
            Warning('batch size should not be larger than max memory size. Setting batch size to memory size')
            batch_size = self.memory.maxlen

        batch_size = min(batch_size, len(self.memory))

        inputs = []
        targets = []

        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size)
        random.shuffle(batch)
        for state, action, next_state, reward in batch:
            inputs.append(state)
            target = self.Q.predict(state)[0]

            # debug, "this should never happen"
            assert not np.array_equal(state, next_state)

            # non-zero reward indicates terminal state
            if reward:
                target[action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                Q_sa = np.max(self.Q.predict(next_state)[0])
                target[action] = reward + self.discount * Q_sa
            targets.append(target)

        # to numpy matrices
        return np.vstack(inputs), np.vstack(targets)

    def replay(self, batch_size):
        inputs, targets = self._prep_batch(batch_size)
        loss = self.Q.train_on_batch(inputs, targets)
        return loss

    def save(self, fname):
        self.Q.save_weights(fname)

    def load(self, fname):
        self.Q.load_weights(fname)
        print(self.Q.get_weights())

if __name__ == "__main__":
    game = Game()
    agent = Agent(game)

    print('training...')
    epochs = 6500
    batch_size = 256
    fname = 'game_weights.h5'

    # keep track of past record_len results
    record_len = 100
    record = deque([], record_len)

    for i in range(epochs):
        game.reset()
        reward = 0
        loss = 0
        # rewards only given at end of game
        while reward == 0:
            # game.render()
            prev_state = game.state
            action = agent.choose_action()
            game.move(action)
            reward = game.update()
            new_state = game.state

            # debug, "this should never happen"
            assert not np.array_equal(new_state, prev_state)

            agent.remember(prev_state, action, new_state, reward)
            loss += agent.replay(batch_size)

        # if running in a terminal, use these instead of print:
        sys.stdout.flush()
        sys.stdout.write('epoch: {:04d}/{} | loss: {:.3f} | win rate: {:.3f}\r'.format(i+1, epochs, loss, sum(record)/len(record) if record else 0))
        # if i % 100 == 0:
            # print('epoch: {:04d}/{} | loss: {:.3f} | win rate: {:.3f}\r'.format(i+1, epochs, loss, sum(record)/len(record) if record else 0))

        record.append(reward if reward == 1 else 0)

    agent.save(fname)
