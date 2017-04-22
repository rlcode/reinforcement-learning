import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 5000


# this is DQN Agent for the MountainCar
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MountainCar learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.1
        self.batch_size = 64
        self.train_start = 100000
        # create replay memory using deque
        self.memory = deque(maxlen=100000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * \
                                          np.amax(self.target_model.predict(next_state)[0])
            update_input[i] = state
            update_target[i] = target

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # in case of CartPole-v1, you can play until 500 time step
    env = gym.make('MountainCar-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = 2 # env.action_space.n
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    action_fake = 0
    goal_position = 0.5
    global_step = 0

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        at_top = False
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            if action == 0:
                action_fake = 0
            if action == 1:
                action_fake = 2

            next_state, reward, done, info = env.step(action_fake)
            next_state = np.reshape(next_state, [1, state_size])

            if next_state[0][0] >= goal_position:
                reward = 100
                at_top = True
            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_replay()
            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/MountainCar_DQN1.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  global_step:", global_step, "  epsilon:", agent.epsilon, "  at_top:", at_top)

        # save the model
        # if e % 50 == 0:
        #     agent.save_model("./save_model/MountainCar_DQN.h5")