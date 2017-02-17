import copy
import numpy as np
import pygame
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from ple import PLE
from ple.games.snake import Snake
from pygame.constants import K_a, K_s, K_w, K_d


EPISODE = 10000

def process_state(state):
    return np.array([state.values()])

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory= deque(maxlen = 10000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.learning_rate = 0.005
        self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim = 5, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(5, activation='linear', init='he_uniform'))

        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([K_a, None, K_w, K_s, K_d])
        act_values = self.model.predict(state)
        return  [K_a, None, K_w, K_s, K_d][np.argmax(act_values[0])]

    def replay(self, batch_size):
        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batchs, replace= False)
        for i in batchs:
            state, action, reward, next_state = self.memory[i]
            target = reward + self.gamma * \
                                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
            if self.epsilon > self.epsilon_min:
               self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    game = Snake()
    env = PLE(game, display_screen= True, state_preprocessor=process_state)
    agent = DQNAgent(env)
    #agent.load('./save/catcher.h5')


    for e in range(EPISODE):
        env.init()
        score = 0
        state = game.getGameState()

        import operator
        state = sorted(state.items(), key=operator.itemgetter(0))
        for i in range(len(state)):
            state[i] = state[i][1]
        state[2] = len(state[2])
        state = np.array([state])
        for time_t in range(5000):
            action = agent.act(state)

            reward = env.act(action)
            score += reward

            next_state = game.getGameState()
            next_state = sorted(next_state.items(), key=operator.itemgetter(0))
            for i in range(len(next_state)):
                next_state[i] = next_state[i][1]
            next_state[2] = len(next_state[2])
            #print (next_state[2])
            next_state = np.array([next_state])

            action = [K_a, None, K_w, K_s, K_d].index(action)

            agent.remember(state, action, reward, next_state)
            state = copy.deepcopy(next_state)

            if env.game_over():
                print("episode: {}/{}, score: {}, memory size: {}, e: {}"
                      .format(e, EPISODE, score,
                              len(agent.memory), agent.epsilon))
                break

            if e % 100 == 0:
                agent.save("./save/catcher.h5")

            agent.replay(16)




















