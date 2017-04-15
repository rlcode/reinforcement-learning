from waterworld import WaterWorld
from ple import PLE
import numpy as np
from pygame.constants import K_w, K_a, K_s, K_d
from dqn_agent import DQNAgent
import copy

game = WaterWorld(width=256, height=256, num_creeps=10)
p = PLE(game, fps=30, display_screen=True)
actions = [K_w, K_a, K_d, K_s, None]
p.init()
reward = 0.0
episodes = 2000
agent = DQNAgent()
nb_frames = 2000

# scores = []
# time = []

agent = DQNAgent()
agent.load("saved_nets_3 : 300")

# Iterate the game
for e in range(episodes):

    p.reset_game()
    state = np.reshape(game.getGameState(), [1, 52])
    tick = 0

    while tick <= 3000:
        tick += 1

        action = agent.act(state)
        reward = p.act(actions[action])
        # print("action : " , action)
        # print("state : ", state)
        next_state, done = game.getGameState(), p.game_over()
        next_state = np.reshape(next_state, [1, 52])

        agent.remember(state, action, reward, next_state, done)

        state = copy.deepcopy(next_state)

        if tick % 30 == 0:
            agent.replay(32)

        if p.game_over() or tick == 3000:

            score = game.getScore()
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}, epsilon : {}"
                  .format(e, episodes, score, agent.epsilon))
            # scores.append(score)
            # time.append(e)
            break
            # train the agent with the experience of the episode

    if e % 50 == 0:
        # plt.subplot(211)
        # plt.plot(time, scores, 'b-')
        # plt.savefig("./waterworld_DQN.png")
        agent.save("saved_nets_4 : " + str(e))
