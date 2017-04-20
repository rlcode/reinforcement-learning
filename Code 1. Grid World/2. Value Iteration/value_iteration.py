# -*- coding: utf-8 -*-
import copy
import random

DISCOUNT_FACTOR = 0.9


class ValueIteration:
    def __init__(self, env):
        # environment object
        self.env = env
        # creaking 2 dimension list for the value function
        self.value_table = [[0.00] * env.width for _ in range(env.height)]

    # get next value function table from the current value function table
    def iteration(self):
        value_table_copy = copy.deepcopy(self.value_table)
        for state in self.env.get_all_states():
            value_table_copy[state[0]][state[1]] = round(self.calculate_max_value(state), 2)
        self.value_table = copy.deepcopy(value_table_copy)
        print("value_table  : " , self.value_table)

    # calculate next value function using Bellman Optimality Equation
    def calculate_max_value(self, state):

        if state == [2, 2]:
            return 0.0

        # empty list for the value function
        value_list = []

        # do the calculation for the all possible actions
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value_list.append((reward + DISCOUNT_FACTOR * next_value))

        print("value _ list : " , value_list)

        # return the maximum value(it is optimality equation!!)
        return max(value_list)

    # get action according to the current value function table
    def get_action(self, state, random_pick=True):

        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # calculating q values for the all actions and
        # append the action to action list which has maximum q value
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + DISCOUNT_FACTOR * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        # pick one action from action_list which has same q value
        if random_pick is True:
            return random.sample(action_list, 1)[0]

        return action_list

    def get_value_table(self):
        return copy.deepcopy(self.value_table)

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)
