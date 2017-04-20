from environment import Env
from QLearning_agent import QLearningAgent


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # reset environment and initialize state
        state = env.reset()

        while True:
            env.render()

            # get action of state from agent
            action = agent.get_action(str(state))

            # take action and doing one step in the environment
            # environment return next state, immediate reward and
            # information about terminal of episode
            next_state, reward, done = env.step(action)

            # with sample <s,a,r,s'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state

            # print q function of all states at screen
            env.print_value_all(agent.q_table)

            # if episode ends, then break
            if done:
                break
