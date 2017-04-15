import os
import numpy as np
from ple import PLE
from ple.games.catcher import Catcher
from pygame.constants import K_a, K_d
from agent import DQNAgent
from agent import process_state
import pylab

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

EPISODES = 100000
np.random.seed(0)

if __name__ == "__main__":
	game = Catcher(width=320, height=320)
	env = PLE(game, display_screen=True, state_preprocessor=process_state)
	agent = DQNAgent(env)
	# agent.load("./save/catcher.h5")

	# 초기화
	# pylab.title("reward")
	# pylab.xlabel("episodes")
	# pylab.ylabel("rewards")
	env.init()
	scores, time = [], []
	for e in range(EPISODES):

		env.reset_game()
		state = env.getGameState()
		state = np.array([list(state[0])])
		score = 0
		for time_t in range(20000):
			action = agent.act(state)

			reward = env.act(action)  # 액션 선택
			score += reward

			next_state = env.getGameState()
			next_state = np.array([list(next_state[0])])

			action = [K_a, None, K_d].index(action)

			agent.remember(state, action, reward, next_state, env.game_over())
			state = next_state

			if env.game_over() or time_t == 19999:
				# 에피소드가 끝나면 출력
				print("episode: {}/{}, score: {}, memory size: {}, e: {}"
				      .format(e, EPISODES, score,
				              len(agent.memory), agent.epsilon))

				# 리워드 플랏을 위한 코드
				scores.append(score)
				time.append(e + 1)
				if e % 10 == 0:
					pylab.plot(time, scores, 'b')
					pylab.savefig("./save/catcher_dqn_1.png")
				break

			if e % 100 == 0:

				agent.save("./save/catcher_1.h5")

			if time_t % 4 == 3:
				agent.replay(32)
