import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gridworld import GraphicDisplay, PolicyEnv as Env  # noqa: E402

class ValueIteration:
    def __init__(self, env):
        self.env = env
        # 2-d list for the value function
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    # get next value function table from the current value function table
    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width
                                    for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            value_list = []

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))
            # return the maximum value(it is the optimality equation!!)
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    # get action according to the current value function table
    def get_action(self, state):
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
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    vi = ValueIteration(env)
    display_ref = {"display": None}

    def on_calculate():
        vi.value_iteration()
        display_ref["display"].show_values(vi.value_table)

    def on_print_policy():
        # Build a policy table from the greedy actions implied by V.
        policy = [[[0.0] * 4 for _ in range(env.width)] for _ in range(env.height)]
        for x in range(env.height):
            for y in range(env.width):
                if [x, y] == [2, 2]:
                    continue
                actions = vi.get_action([x, y])
                if not actions:
                    continue
                p = 1.0 / len(actions)
                for a in actions:
                    policy[x][y][a] = p
        display_ref["display"].show_arrows(policy)

    def on_move():
        display_ref["display"].move_along_policy(vi.get_action)

    def on_clear():
        vi.value_table = [[0.0] * env.width for _ in range(env.height)]
        display_ref["display"].clear()
        display_ref["display"].agent_pos = [0, 0]

    display = GraphicDisplay(
        vi,
        title="Value Iteration",
        buttons=[
            ("Calculate", on_calculate),
            ("Print Policy", on_print_policy),
            ("Move", on_move),
            ("Clear", on_clear),
        ],
    )
    display_ref["display"] = display
    display.mainloop()
