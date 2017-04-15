from waterworld import WaterWorld
from ple import PLE
from pygame.constants import K_w, K_a, K_s, K_d
from ddqn_agent import DQNAgent
import copy
import numpy as np
import matplotlib.pyplot as plt

game = WaterWorld(width=256, height=256, num_creeps=8)
p = PLE(game, display_screen=False)
actions = [K_w, K_a, K_d, K_s, None]
p.init()
reward = 0.0
episodes = 20000
agent = DQNAgent()
nb_frames = 20000

plt.ion()
plt.title("reward")
plt.xlabel("episodes")
plt.ylabel("rewards")
scores, time = [], []


def dic_to_list(state):
    temp_state = []

    for key, value in state.items():
        if key == 'creeps':
            for input in value:
                temp_state.append(input)
        else:
            temp_state.append(value)
    return temp_state

agent = DQNAgent()
#agent.load('save_net_changed')
global_step = 0
for e in range(episodes):

    p.reset_game()
    state = dic_to_list(game.getGameState())
    state = np.reshape(state, [1, 26])
    ex_reward = 0
    for _ in range(3000):
        global_step += 1
        action = agent.act(state)
        reward = p.act(actions[action])

        next_state, done = dic_to_list(game.getGameState()), (p.game_over() or _ == 2999)
        next_state = np.reshape(next_state, [1, 26])

        agent.remember(state, action, reward, next_state, done)

        state = copy.deepcopy(next_state)
        if _ % 4 == 0:
            agent.replay(32)

        if global_step % 500 == 0:
            agent.update_target_model()

        if p.game_over():
            print("episode: {}/{}, score: {}"
                  .format(e, episodes, game.getScore()))

            scores.append(game.getScore())
            time.append(e + 1)
            if e % 10 == 0:
                plt.plot(time, scores, 'b-')
                plt.pause(0.000001)

            break

    if not p.game_over():
        scores.append(game.getScore())
        time.append(e + 1)
        if e % 10 == 0:
            plt.plot(time, scores, 'b-')
            plt.savefig('waterworld_ddqn.png')
            plt.pause(0.000001)

        print("episode: {}/{}, score: {}, e: {}"
          .format(e, episodes, game.getScore(), agent.epsilon))
        #agent.save('save_net_changed')
