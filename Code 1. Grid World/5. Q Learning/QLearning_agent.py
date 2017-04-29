import numpy as np
import random


class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = {}

        # check whether the state was visited
        # if this is first visitation, then initialize the q function of the state

    def check_state_exist(self, state):
        if state not in self.q_table.keys():
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        self.check_state_exist(next_state)
        q_1 = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        self.check_state_exist(state)

        if np.random.rand() > self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
