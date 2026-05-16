from gridworld import GraphicDisplay, PolicyEnv as Env


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
    value_iteration = ValueIteration(env)
    display = GraphicDisplay(value_iteration, title="Value Iteration")

    def on_calculate():
        value_iteration.value_iteration()
        display.show_values(value_iteration.value_table)

    def on_print_policy():
        # Build a policy arrow table from the greedy actions implied by V.
        policy = [[[0.0] * 4 for _ in range(env.width)] for _ in range(env.height)]
        for state in env.get_all_states():
            x, y = state
            actions = value_iteration.get_action(state)
            if not actions:
                continue
            prob = 1.0 / len(actions)
            for a in actions:
                policy[x][y][a] = prob
        display.show_arrows(policy)

    def on_move():
        display.move_along_policy(value_iteration.get_action)

    def on_clear():
        value_iteration.__init__(env)
        display.clear()
        display.agent_pos = [0, 0]
        display.clicks.clear()

    display.buttons = [
        ("Calculate",    on_calculate),
        ("Print Policy", on_print_policy, lambda: display.click_count("Calculate") > 0),
        ("Move",         on_move,         lambda: display.click_count("Print Policy") > 0),
        ("Clear",        on_clear),
    ]
    display.mainloop()
