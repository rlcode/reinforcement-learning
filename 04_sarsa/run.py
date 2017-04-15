from environment import Env
from agent import Sarsa


def update():
    for episode in range(1000):
        # 환경 초기화와  환경으로 부터 현재 상태 받아오기
        state = env.reset()

        # 에이전트로부터 해당 상태에 대한 행동을 받아옴
        action = agent.get_action(str(state))

        while True:
            # Gui 렌더링
            env.render()

            # 에이전트의 행동을 취하고 다음 상태와 보상과 에피소드가 끝났는지의 여부를 받아옴
            state_, reward, done = env.step(action)

            # 에이전트로부터 해당 상태에 대한 행동을 받아옴
            action_ = agent.get_action(str(state_))

            # 에이전트의 learn함수에 S A R S_ A_ 를 넣어줌
            agent.learn(str(state), action, reward, str(state_), action_)

            # 현재 상태에 다음 상태를 대입, 현재 행동에 다음 행동을 대입
            state = state_
            action = action_

            env.print_value_all(agent.q_table)

            # 에피소드가 끝나면 break
            if done:
                break

    # 모든 에피소드가 다 끝나면 게임오버
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    env = Env()
    agent = Sarsa(actions=list(range(env.n_actions)))
    update()