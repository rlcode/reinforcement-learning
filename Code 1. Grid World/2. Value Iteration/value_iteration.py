# -*- coding: utf-8 -*-
import random
from environment import GraphicDisplay, Env

DISCOUNT_FACTOR = 0.9


class ValueIteration:
    def __init__(self, env):
        # environment object
        self.env = env
        # creaking 2 dimension list for the value function
        self.value_table = [[0.00] * env.width for _ in range(env.height)]

    # get next value function table from the current value function table
    def value_iteration(self):
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # empty list for the value function
            value_list = []

            # do the calculation for the all possible actions
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + DISCOUNT_FACTOR * next_value))
            # return the maximum value(it is optimality equation!!)
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

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

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
