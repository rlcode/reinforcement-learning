# -*- coding: utf-8 -*-
import random
from environment import GraphicDisplay, Env

DISCOUNT_FACTOR = 0.9


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        # 2-d list for the value function
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # list of random policy (same probability of up, down, left, right)
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                    for _ in range(env.height)]
        # setting terminal state
        self.policy_table[2][2] = []

    def policy_evaluation(self):
        next_value_table = [[0.00] * self.env.width
                                    for _ in range(self.env.height)]

        # Bellman Expectation Equation for the every states
        for state in self.env.get_all_states():
            value = 0.0
            # keep the value function of terminal states as 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += self.get_policy(state, action) * \
                        (reward + DISCOUNT_FACTOR * next_value)

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            value = -99999
            max_index = []
            result = [0.0, 0.0, 0.0, 0.0] # initialize the policy

            # for every actions, calculate
            # [reward + (discount factor) * (next state value function)]
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + DISCOUNT_FACTOR * next_value

                # We normally can't pick multiple actions in greedy policy.
                # but here we allow multiple actions with same max values
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

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    def get_action(self, state):
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # return the action in the index
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return self.env.possible_actions[index]

    def get_policy(self, state, action=None):
        # if no action is given, then return the probabilities of all actions
        if action is None:
            return self.policy_table[state[0]][state[1]]
        if state == [2, 2]:
            return 0.0
        action_index = self.env.possible_actions.index(action)
        return self.policy_table[state[0]][state[1]][action_index]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
