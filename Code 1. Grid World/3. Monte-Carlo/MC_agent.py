import numpy as np
import pandas as pd


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
        self.value_table = pd.DataFrame(columns=['value'])

    # check whether the state was visited
    # if this is first visitation, then initialize the q function of the state
    def check_state_exist(self, state):
        if str(state) not in self.value_table.index:
            self.value_table = self.value_table.append(
                pd.Series(
                    [0] * len(self.value_table.columns),
                    index=self.value_table.columns,
                    name=str(state)
                )
            )

    # append sample to memory(state, reward, done)
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # for every episode, agent updates q function of visited states
    def update(self):
        G_t = 0
        visit_state = []
        for reward in reversed(self.samples):
            state = str(reward[0])
            if state not in visit_state:
                visit_state.append(state)
                G_t = self.discount_factor * (reward[1] + G_t)
                self.check_state_exist(state)
                value = self.value_table.ix[state, 'value']
                self.value_table.ix[state, 'value'] = value + self.learning_rate * (G_t - value)
                print("state : ", state, " G : ", G_t, " update : ", value + self.learning_rate * (G_t - value))
        print("values : ", self.value_table)

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
            next_state = next_state.reindex(np.random.permutation(next_state.index))
            action = next_state.argmax()

        return int(action)

    # get the possible next states
    def possible_next_state(self, state):
        state_col = state[0]
        state_row = state[1]

        next_state = pd.Series(
            [0] * len(self.actions),
            index=self.actions,
        )

        if state_row != 0:
            self.check_state_exist(str([state_col, state_row - 1]))
            next_state.set_value(0, self.value_table.ix[str([state_col, state_row - 1]), 'value'])  # up
        if state_row != self.height - 1:
            self.check_state_exist(str([state_col, state_row + 1]))
            next_state.set_value(1, self.value_table.ix[str([state_col, state_row + 1]), 'value'])  # down
        if state_col != 0:
            self.check_state_exist(str([state_col - 1, state_row]))
            next_state.set_value(2, self.value_table.ix[str([state_col - 1, state_row]), 'value'])  # left
        if state_col != self.width - 1:
            self.check_state_exist(str([state_col + 1, state_row]))
            next_state.set_value(3, self.value_table.ix[str([state_col + 1, state_row]), 'value'])  # right

        return next_state
