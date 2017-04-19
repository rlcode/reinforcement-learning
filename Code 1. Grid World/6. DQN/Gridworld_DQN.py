import copy
import pylab
import random
import numpy as np
from environment import Env
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000


class DQNAgent:
    def __init__(self):
        self.render = False

        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 22
        self.discount_factor = 0.99  # decay rate
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.train_start = 1000

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memory = deque(maxlen=10000)
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep Q Learning
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(5, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.choice(self.action_space)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = np.random.choice(len(self.memory), batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            reward = np.float32(reward)
            state = np.float32(state)
            next_state = np.float32(next_state)
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target = reward + self.discount_factor * \
                                  np.amax(self.model.predict(next_state)[0])

            update_input[i] = state
            update_target[i] = target

            # Train the Neural Net with the state and target_f
            self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # maze game
    # env = Maze()
    env = Env()
    agent = DQNAgent()

    global_step = 0
    # agent.load("same_vel_episode2 : 1000")
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 22])

        while not done:
            # fresh env
            if agent.render:
                env.render()
            global_step += 1

            # RL choose action based on observation and go one step
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 22])

            agent.replay_memory(state, action, reward, next_state, done)
            # every time step we do train from the replay memory
            agent.train_replay()
            score += reward
            # swap observation
            state = copy.deepcopy(next_state)
            print("reward:", reward, "  time_step:", global_step, "  epsilon:", agent.epsilon)

            if global_step % 100 == 0:
                agent.update_target_model()

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/10by10.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

        if e % 100 == 0:
            agent.save_model("./save_model/10by10 : " + str(e))

    # end of game
    print('game over')
    env.destroy()




