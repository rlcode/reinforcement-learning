import numpy as np
import pandas as pd


class Sarsa:
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, e_greedy=1):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = e_greedy
        # DataFrame 사용법 참고 바람
        self.q_table = pd.DataFrame(columns=self.actions)

    # 예전에 가본 state 인지 아닌 지 판별하고 안가본 state 라면 초기화
    def check_state_exist(self, state):
        # 안가본 state 일 때만 초기화
        if state not in self.q_table.index:
            # 새로운 state 를 q_table 에 추가
            # 초기화는 [0, 0, 0, 0]으로
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # 큐 함수를 살사 알고리즘에 따라 업데이트
    def learn(self, s, a, r, s_, a_):
        # 먼저 가본 적이 있는 상태인지 확인하고 아니라면 초기화
        self.check_state_exist(s_)
        self.q_table.ix[s, a] = \
            self.q_table.ix[s, a] + \
            self.lr * (r + self.gamma * self.q_table.ix[s_, a_] - self.q_table.ix[s, a])

    # 현재 상태에 대해 행동을 받아오는 함수
    def get_action(self, state):
        self.check_state_exist(state)
        # epsilon 보다 rand 함수로 뽑힌 수가 작으면 큐 함수에 따른 행동 리턴
        if np.random.rand() < self.epsilon:
            # 최적의 행동 선택
            state_action = self.q_table.ix[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()

        # epsilon 보다 rand 함수로 뽑힌 수가 크면 랜덤으로 행동을 리턴
        else:
            # 임의의 행동을 선택
            action = np.random.choice(self.actions)
        return action
