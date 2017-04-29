from environment import Env
from SARSA_agent_edit import SARSAgent

if __name__ == "__main__":
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # reset environment and initialize state
        state = env.reset()
        # get action of state from agent
        action = agent.get_action(str(state))

        while True:
            env.render()

            # take action and doing one step in the environment
            # environment return next state, immediate reward and
            # information about terminal of episode
            next_state, reward, done = env.step(action)

            # get action of state from agent
            next_action = agent.get_action(str(next_state))

            # with sample <s,a,r,s',a'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            # print q function of all states at screen
            env.print_value_all(agent.q_table)

            # if episode ends, then break
            if done:
                break