from environment import Env
from agent import DQNAgent
import numpy as np
import copy


def run():
    step = 0
    episodes = 2000
    # agent.load("same_vel_episode2 : 1000")
    for e in range(episodes):
        # initial observation
        state = env.reset()
        state = np.reshape(state, [1, 22])
        score = 0
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = agent.act(state)

            # RL take action and get next observation and reward
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 22])

            agent.remember(state, action, reward, next_state, done)
            score += reward

            # if (step > 200) and (step % 5 == 0):
            #     agent.replay(32)

            # swap observation
            state = copy.deepcopy(next_state)
            # break while loop when end of this episode
            if done:
                print("episode: {}/{}, score: {},epsilon : {}"
                      .format(e, episodes, score, agent.epsilon))
                break
            step += 1
        agent.replay(32)
        if e % 100 == 0:
            agent.save("10by10 : " + str(e))


    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    # env = Maze()
    env = Env()
    agent = DQNAgent()
    env.after(1000, run)
    env.mainloop()
