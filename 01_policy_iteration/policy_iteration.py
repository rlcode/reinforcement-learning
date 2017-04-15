# -*- coding: utf-8 -*-
import copy
import random

DISCOUNT_FACTOR = 0.9  # 감가율

class PolicyIteration:
    def __init__(self, env):
        # 환경에 대한 객체
        self.env = env
        # 가치함수의 값을 담을 2차원 리스트 생성 및 초기화
        self.value_table = [[0.00] * env.width for _ in range(env.height)]
        # 정책을 담을 리스트 생성
        # 상, 하, 좌, 우 동일한 확률로 움직이는 무작위 정책
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        # 마침 상태의 설정
        self.policy_table[2][2] = []

    # 정책 평가
    def policy_evaluation(self):
        # 현재 가치함수를 메모리에 저장
        next_value_table = copy.deepcopy(self.value_table)
        # 모든 상태들에 대해 벨만 기대 방정식 계산
        for state in self.env.get_all_states():
            next_value_table[state[0]][state[1]] = round(self.calculate_value(state), 2)
        # 가치함수 업데이트
        self.value_table = copy.deepcopy(next_value_table)

    # 벨만 기대 방정식
    def calculate_value(self, state):
        value = 0
        # 벨만 기대방정식에 따라 가치함수의 값을 계산
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value += self.get_policy(state, action) * (reward + DISCOUNT_FACTOR*next_value)
        # 마침 상태의 가치함수 = 0
        if state == [2, 2]:
            return 0.0

        return value

    # 탐욕 정책 발전
    def greedy_policy(self, state):

        value = -99999
        max_index = []
        # 순서대로 상 하 좌 우 행동을 취할 확률
        result = [0.0, 0.0, 0.0, 0.0]

        # 모든 행동들에 대해서 보상 + (감가율 * 다음 상태 가치함수)을 계산
        for index, action in enumerate(self.env.possible_actions):
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            temp = reward + DISCOUNT_FACTOR*next_value

            # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
            if temp == value:
                max_index.append(index)
            elif temp > value:
                value = temp
                max_index.clear()
                max_index.append(index)

        # 행동의 확률을 계산
        prob = 1 / len(max_index)

        for index in max_index:
            result[index] = prob

        return result

    # 정책 발전
    def policy_improvement(self):
        next_policy = self.get_policy_table()
        for state in self.env.get_all_states():
            # 최종 상태는 제외
            if state == [2, 2]:
                continue
            # 탐욕 정책 발전
            next_policy[state[0]][state[1]] = self.greedy_policy(state)
        self.policy_table = next_policy

    # 특정 상태에서 정책에 따른 행동
    def get_action(self, state):
        # 0~1 사이의 값을 랜덤으로 추출
        random_pick = random.randrange(100) / 100
        # 현재 상태의 정책
        policy = self.get_policy(state)
        policy_sum = 0.0
        # 정책에 담긴 확률 값을 하나하나 더해가면서 random_pick을 넘어가는 순간의 index에 해당하는 행동을 리턴
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return self.env.possible_actions[index]

    # 전체 정책 리스트 받아오기
    def get_policy_table(self):
        return copy.deepcopy(self.policy_table)

    # 상태와 행동에 따른 정책 받아오기
    def get_policy(self, state, action=None):
        # 행동이 없으면 4개의 행동에 대한 확률 전체 리스트를 반환
        if action is None:
            return self.policy_table[state[0]][state[1]]
        # 최종 상태는 제외
        if state == [2, 2]:
            return 0.0
        # 현재 상태의 행동을 반환
        return self.policy_table[state[0]][state[1]][self.env.possible_actions.index(action)]

    # 전체 가치함수 리스트 받아오기
    def get_value_table(self):
        return copy.deepcopy(self.value_table)

    # 특정 상태의 가치함수를 반환
    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)
