import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 1000


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.01
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.actor, self.critic = self.build_model()

        # method for training the model
        self.actor_optimizer = self.actor_optimizer()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    def build_model(self):
        # make actor network
        actor = Sequential()
        actor.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        actor.summary()

        # make critic network
        critic = Sequential()
        critic.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer="he_uniform"))
        critic.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        critic.summary()

        return actor, critic

    # make loss function for the Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        good_prob = K.sum(action * self.actor.output, axis=1)
        eligibility = K.log(good_prob) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, discounted_rewards], [], updates=updates)
        return train

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy network every episode
    def train_episodes(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.critic.fit(np.array(self.states), discounted_rewards, epochs=1, verbose=0)

        self.actor_optimizer([self.states, self.actions, advantages])
        self.states, self.actions, self.rewards = [], [], []

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make Policy Gradient agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # agent.load_model("./save_model/Cartpole_A2C.h5")

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # save the sample <s, a, r> to the memory
            agent.memory(state, action, reward)

            score += reward
            state = next_state

            if done:
                # every episode, agent learns from sample returns
                agent.train_episodes((score!=500))

                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Cartpole_A2C.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.save_model("./save_model/Cartpole_A2C")
