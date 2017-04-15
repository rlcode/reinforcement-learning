# -*- coding: utf-8 -*-
import copy
import random

DISCOUNT_FACTOR = 0.9  # 감가율


class ValueIteration:
    def __init__(self, env):
        self.env = env
        # 가치(value)값을 담을 2차원 리스트
        self.value_table = [[0.00] * env.width for _ in range(env.height)]
        # 계산을 위한 가치(value)리스트의 복사본

    # 모든 상태에 대한 가치 값을 계산 하는 함수입니다
    def iteration(self):
        value_table_copy = copy.deepcopy(self.value_table)
        for state in self.env.get_all_states():
            value_table_copy[state[0]][state[1]] = round(self.calculate_max_value(state), 2)
        self.value_table = copy.deepcopy(value_table_copy)
        print("value_table  : " , self.value_table)

    # 벨만 최적 방정식을 계산
    def calculate_max_value(self, state):
        # 최종 상태는 제외
        if state == [2, 2]:
            return 0.0

        value_list = []

        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value_list.append((reward + DISCOUNT_FACTOR * next_value))

        print("value _ list : " , value_list)
        # value_list 에서 가장 큰 값을 리턴한다
        return max(value_list)

    def get_action(self, state, random_pick=True):

        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # 모든 행동들에 대해 큐 값을 계산하여 최대가 되는 행동(복수라면 모두)을 action_list 에 담는다
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + DISCOUNT_FACTOR * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        # action_list 중에서 랜덤으로 하나를 리턴한다.
        if random_pick is True:
            return random.sample(action_list, 1)[0]

        return action_list

    # 전체 가치(value)값 리스트 받아오기
    def get_value_table(self):
        return copy.deepcopy(self.value_table)

    # 특정 상태(state)의 가치(value)를 반환하는 함수
    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)
