import numpy as np
import pandas as pd
import ast


class MonteCarlo:
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, e_greedy=0.9, height=5, width=5):
        # actions = [0, 1, 2, 3]
        self.width = width
        self.height = height
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = e_greedy
        # DataFrame 사용법 참고 바람
        self.returns = []
        self.value_table = pd.DataFrame(columns=['value'])

    # 예전에 가본 state 인지 아닌 지 판별하고 안가본 state 라면 초기화
    def check_state_exist(self, s):
        if str(s) not in self.value_table.index:
            self.value_table = self.value_table.append(
                pd.Series(
                    [0] * len(self.value_table.columns),
                    index=self.value_table.columns,
                    name=str(s)
                )
            )

    def stack_returns(self, s, r, done):
        self.returns.append([s, r, done])

    def update(self):
        G = 0
        visit_s = []
        for r in reversed(self.returns):
            s = str(r[0])
            if s not in visit_s:
                visit_s.append(s)
                G = self.gamma * (r[1] + G)
                self.check_state_exist(s)
                V = self.value_table.ix[s, 'value']
                self.value_table.ix[s, 'value'] = V + self.alpha * (G - V)
                print("state : ", s, " G : ", G, " update : ", V + self.alpha * (G - V))
        print("values : ", self.value_table)

    # 현재 상태에 대해 행동을 받아오는 함수
    def get_action(self, state):
        self.check_state_exist(state)
        # epsilon 보다 rand 함수로 뽑힌 수가 작으면 큐 함수에 따른 행동 리턴
        if np.random.rand() < self.epsilon:
            # 최적의 행동 선택
            state_ = self.possible_next_s(state)
            state_ = state_.reindex(np.random.permutation(state_.index))
            action = state_.argmax()

        # epsilon 보다 rand 함수로 뽑힌 수가 크면 랜덤으로 행동을 리턴
        else:
            # 임의의 행동을 선택
            action = np.random.choice(self.actions)
        return int(action)

    def possible_next_s(self, s):

        col = s[0]
        row = s[1]

        s_ = pd.Series(
            [0] * len(self.actions),
            index=self.actions,
        )

        if row != 0:
            self.check_state_exist(str([col, row - 1]))
            s_.set_value(0, self.value_table.ix[str([col, row - 1]), 'value'])  # up
        if row != self.height - 1:
            self.check_state_exist(str([col, row + 1]))
            s_.set_value(1, self.value_table.ix[str([col, row + 1]), 'value'])  # down
        if col != 0:
            self.check_state_exist(str([col - 1, row]))
            s_.set_value(2, self.value_table.ix[str([col - 1, row]), 'value'])  # left
        if col != self.width - 1:
            self.check_state_exist(str([col + 1, row]))
            s_.set_value(3, self.value_table.ix[str([col + 1, row]), 'value'])  # right

        return s_
