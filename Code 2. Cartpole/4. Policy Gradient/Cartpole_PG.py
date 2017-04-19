import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 1000


class PGAgent:
    def __init__(self, state_size, action_size):
        # Cartpole이 학습하는 것을 보려면 True로 바꿀 것
        self.render = True
        
        # agent를 학습시키지 않으려면 False로 바꿀 것
        self.is_train = True

        # state와 action의 크기를 가져와서 모델을 생성하는데 사용함
        self.state_size = state_size
        self.action_size = action_size

        # Cartpole REINFORCE 학습의 Hyper parameter 들
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 학습할 모델을 생성
        self.model = self.build_model()

        # Policy Gradient 네트워크 학습하는 함수를 만듬
        self.optimizer = self.optimizer()

        # 상태, 행동, 보상을 기억하기 위한 리스트 생성
        self.states, self.actions, self.rewards = [], [], []

    # Deep Neural Network 를 통해서 정책을 근사
    # 상태가 입력, 각 행동에 대한 확률이 출력인 모델을 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))

        # 마지막 softmax 계층으로 각 행동에 대한 확률을 만드는 모델을 생성
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()

        return model

    def optimizer(self):
        action = K.placeholder(shape=[None, 2])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Policy Gradient 의 핵심
        # log(정책) * return 의 gradient 를 구해서 최대화시킴
        good_prob = K.sum(action * self.model.output, axis=1)
        eligibility = K.log(good_prob) * discounted_rewards
        loss = -K.sum(eligibility)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

        return train

    # 행동의 선택은 현재 네트워크에 대해서 각 행동에 대한 확률로 정책을 사용
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 에피소드가 끝나면 해당 에피소드의 보상를 이용해 return을 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 각 스텝의 <s, a, r>을 저장하는 함수
    def memory(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 에피소드가 끝나면 모아진 메모리로 학습
    def train_episodes(self):
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    # 저장한 모델을 불러옴
    def load_model(self, name):
        self.model.load_weights(name)

    # 학습된 모델을 저장함
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # CartPole-v1의 경우 500 타임스텝까지 플레이가능
    env = gym.make('CartPole-v1')

    # 환경으로부터 상태와 행동의 크기를 가져옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # PG 에이전트의 생성
    agent = PGAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # agent.load_model("./save_model/cartpole-master.h5")

        while not done:
            if agent.render:
                env.render()

            # 현재 상태에서 행동을 선택하고 한 스텝을 진행
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score==499 else -100

            # <s, a, r>을 memory에 저장
            if agent.is_train:
                agent.memory(state, action, reward)

            score += reward
            state = next_state

            if done:
                env.reset()
                # 매 에피소드마다 모아온 <s, a, r>을 학습
                agent.train_episodes()

                # 에피소드에 따른 score를 plot
                score = score if score==500 else score+100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Cartpole_PG.png")
                print("episode:", e, "  score:", score)
                
                # 지난 10 에피소드의 평균이 490 이상이면 학습을 멈춤
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.is_train = False

        # 50 에피소드마다 학습 모델을 저장
        if e % 50 == 0:
            agent.save_model("./save_model/Cartpole_PG.h5")
