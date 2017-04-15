from environment import Env
from agent import MonteCarlo

"""
로그를 찍어보니 첫 state가 update 에서 빠지는 경우가 있음 디버깅 필요
"""


def update():
    for episode in range(1000):
        # 환경 초기화와  환경으로 부터 현재 상태 받아오기
        state = env.reset()

        # 에이전트로부터 해당 상태에 대한 행동을 받아옴
        action = agent.get_action(state)

        while True:
            # Gui 렌더링
            env.render()

            # 에이전트의 행동을 취하고 다음 상태와 보상과 에피소드가 끝났는지의 여부를 받아옴
            state_, reward, done = env.step(action)

            agent.stack_returns(state_, reward, done)

            # 에이전트로부터 해당 상태에 대한 행동을 받아옴
            action_ = agent.get_action(state_)

            # 에이전트의 learn함수에 S A R S_ A_ 를 넣어줌

            # 현재 상태에 다음 상태를 대입, 현재 행동에 다음 행동을 대입
            action = action_

            # 에피소드가 끝나면 break
            if done:
                print("episode : ", episode)
                print("returns : ", agent.returns)
                agent.update()
                agent.returns.clear()
                break

    # 모든 에피소드가 다 끝나면 게임오버
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    env = Env()
    agent = MonteCarlo(actions=list(range(env.n_actions)))
    update()
