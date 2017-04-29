import numpy as np
import random


# this is Monte-Carlo agent for the grid world
# it learns every episodes from the sample(which is the difference with dynamic programming)
class MCAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.samples = []
        self.value_table = {}

    # check whether the state was visited
    # if this is first visitation, then initialize the q function of the state
    def check_state_exist(self, state):
        if str(state) not in self.value_table.keys():
            self.value_table[str(state)] = 0.0

    # append sample to memory(state, reward, done)
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # for every episode, agent updates q function of visited states
    def update(self):
        G_t = 0
        visit_state = []
        print(self.samples)
        for reward in reversed(self.samples):
            state = str(reward[0])
            if state not in visit_state:
                visit_state.append(state)
                G_t = self.discount_factor * (reward[1] + G_t)
                self.check_state_exist(state)
                value = self.value_table[state]
                self.value_table[state] = value + self.learning_rate * (G_t - value)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        self.check_state_exist(state)

        if np.random.rand() > self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
        return int(action)

    # compute arg_max if multiple candidates exit, pick one randomly
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # get the possible next states
    def possible_next_state(self, state):
        state_col = state[0]
        state_row = state[1]

        next_state = [0.0, 0.0, 0.0, 0.0]

        if state_row != 0:
            self.check_state_exist(str([state_col, state_row - 1]))
            next_state[0] = self.value_table[str([state_col, state_row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if state_row != self.height - 1:
            self.check_state_exist(str([state_col, state_row + 1]))
            next_state[1] = self.value_table[str([state_col, state_row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if state_col != 0:
            self.check_state_exist(str([state_col - 1, state_row]))
            next_state[2] = self.value_table[str([state_col - 1, state_row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if state_col != self.width - 1:
            self.check_state_exist(str([state_col + 1, state_row]))
            next_state[3] = self.value_table[str([state_col + 1, state_row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state
