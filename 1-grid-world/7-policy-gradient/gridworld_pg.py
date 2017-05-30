import copy
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 1000


class PGAgent:
    def __init__(self):
        self.render = False

        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 22
        self.discount_factor = 0.99  # decay rate
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
        # 마지막 softmax 계층으로 각 행동에 대한 확률을 만드는 모델을 생성
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()

        return model

    def optimizer(self):
        action = K.placeholder(shape=[None, 5])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Policy Gradient 의 핵심
        # log(정책) * return 의 gradient 를 구해서 최대화시킴
        good_prob = K.sum(action * self.model.output, axis=1)
        eligibility = K.log(good_prob) * discounted_rewards
        loss = -K.sum(eligibility)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

        return train

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def memory(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    def train_episodes(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # maze game
    # env = Maze()
    env = Env()
    agent = PGAgent()

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

            agent.memory(state, action, reward)
            # every time step we do train from the replay memory
            score += reward
            # swap observation
            state = copy.deepcopy(next_state)

            if done:
                agent.train_episodes()

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/10by10.png")
                print("episode:", e, "  score:", score, "  time_step:", global_step)

        if e % 100 == 0:
            pass
            agent.save_model("./save_model/10by10")

    # end of game
    print('game over')
    env.destroy()