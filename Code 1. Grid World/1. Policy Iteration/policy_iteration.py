# -*- coding: utf-8 -*-
import copy
import random

DISCOUNT_FACTOR = 0.9


class PolicyIteration:
    def __init__(self, env):
        # environment object
        self.env = env
        # creaking 2 dimension list for the value function
        self.value_table = [[0.00] * env.width for _ in range(env.height)]
        # creating list for the policy
        # this is random policy which has same probability for doing up, down, left, right
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        # setting terminal state
        self.policy_table[2][2] = []

    # method for the policy evaluation
    # use Bellman Expectation Equation for calculating next value function table
    def policy_evaluation(self):
        next_value_table = copy.deepcopy(self.value_table)

        # Bellman Expectation Equation for the every states
        for state in self.env.get_all_states():
            next_value_table[state[0]][state[1]] = round(self.calculate_value(state), 2)

        self.value_table = copy.deepcopy(next_value_table)

    # calculating new value function using Bellman Expectation Equation
    def calculate_value(self, state):
        value = 0

        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value += self.get_policy(state, action) * (reward + DISCOUNT_FACTOR * next_value)

        # keep the value function of terminal states as 0
        if state == [2, 2]:
            return 0.0

        return value

    # This is Greedy Policy which always selects action of maximum value
    def greedy_policy(self, state):

        value = -99999
        max_index = []
        # initialize the policy
        result = [0.0, 0.0, 0.0, 0.0]

        # for every actions, calculating [reward + (discount factor) * (next state value function)]
        for index, action in enumerate(self.env.possible_actions):
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            temp = reward + DISCOUNT_FACTOR * next_value

            # For the greedy policy, originally we can't pick multiple actions
            # but in this example, we allow to pick multiple actions which have same maximum values
            if temp == value:
                max_index.append(index)
            elif temp > value:
                value = temp
                max_index.clear()
                max_index.append(index)

        # probability of action
        prob = 1 / len(max_index)

        for index in max_index:
            result[index] = prob

        return result

    # using the greedy policy method, do the policy improvement
    # under the current value function table
    def policy_improvement(self):
        next_policy = self.get_policy_table()
        for state in self.env.get_all_states():

            if state == [2, 2]:
                continue

            next_policy[state[0]][state[1]] = self.greedy_policy(state)
        self.policy_table = next_policy

    # get action according to the current policy
    def get_action(self, state):
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # return the action in the index
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return self.env.possible_actions[index]

    # get the policy table for the all states
    def get_policy_table(self):
        return copy.deepcopy(self.policy_table)

    # get policy of specific state and action
    def get_policy(self, state, action=None):
        # if there is no action given, then return the probabilities of all actions
        if action is None:
            return self.policy_table[state[0]][state[1]]

        if state == [2, 2]:
            return 0.0

        return self.policy_table[state[0]][state[1]][self.env.possible_actions.index(action)]

    def get_value_table(self):
        return copy.deepcopy(self.value_table)

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

