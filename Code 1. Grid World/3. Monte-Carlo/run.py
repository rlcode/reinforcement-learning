from environment import Env
from MC_agent import MCAgent

# main loop
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # reset environment and initialize state
        state = env.reset()

        # take action and doing one step in the environment
        action = agent.get_action(state)

        while True:
            env.render()

            # environment return next state, immediate reward and
            # information about terminal of episode

            next_state, reward, done = env.step(action)

            agent.save_sample(next_state, reward, done)

            # take action and doing one step in the environment
            action = agent.get_action(next_state)

            # at the end of episode, update the q function table
            if done:
                print("episode : ", episode)
                agent.update()
                agent.samples.clear()
                break
