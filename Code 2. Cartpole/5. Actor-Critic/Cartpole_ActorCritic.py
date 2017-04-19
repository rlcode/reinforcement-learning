import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 300


class ACAgent:
    def __init__(self, state_size, action_size):
        # Cartpole이 학습하는 것을 보려면 True로 바꿀 것
        self.render = False

        # state와 action의 크기를 가져와서 모델을 생성하는데 사용함
        self.state_size = state_size
        self.action_size = action_size

        # Cartpole Actor-Critic에 필요한 Hyperparameter들
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.01
        self.batch_size = 32
        self.train_start = 1000
        self.memory = deque(maxlen=10000)

        # Actor-Critic C에 필요한 actor 네트워크와 critic 네트워크를 생성
        self.actor, self.critic = self.build_model()

        # actor 네트워크를 학습시키기 위한 optimizer 를 만듬
        self.actor_optimizer = self.actor_optimizer()

    # Deep Neural Network 를 통해서 정책과 가치를 근사
    # actor -> 상태가 입력, 각 행동에 대한 확률이 출력인 모델을 생성
    # critic -> 상태가 입력, 상태에 대한 가치가 출력인 모델을 생성
    def build_model(self):
        # actor 네트워크 생성
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))

        # critic 네트워크 생성
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer="he_uniform"))
        critic.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, 2])
        advantages = K.placeholder(shape=[None, ])

        # Policy Gradient 의 핵심
        # log(정책) * return 의 gradient 를 구해서 최대화시킴
        good_prob = K.sum(action * self.actor.output, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        loss = -K.sum(eligibility)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)

        return train

    # replay memory에서 batch_size 만큼의 샘플들을 무작위로 뽑아서 학습
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size))
        update_action = np.zeros((self.batch_size, self.action_size))
        update_target = np.zeros((self.batch_size, 1))
        advantages = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            value = self.critic.predict(state)[0]

            # s'의 state value를 가져와서 critic 네트워크를 업데이트함.
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * \
                                  self.critic.predict(next_state)[0]
            update_input[i] = state
            update_action[i] = action
            update_target[i] = target
            advantages[i] = target - value

        # 학습할 정답인 타겟과 현재 자신의 값의 minibatch를 만들고 그것으로 한 번에 critic 모델 업데이트
        self.critic.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0)

        # 상태, 행동, 그에 따른 (target-value)를 넣어 actor 네트워크를 학습함
        self.actor_optimizer([update_input, update_action, advantages])

    # 핻동의 선택은 actor 네트워크에 대해서 각 행동에 대한 확률로 정책을 사용
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 각 스텝의 <s, a, r, s'>을 저장
    def replay_memory(self, state, action, reward, next_state, done):
        act = np.zeros(self.action_size)
        act[action] = 1
        self.memory.append((state, act, reward, next_state, done))

    # 저장한 모델을 불러옴
    def load_model(self, name):
        self.actor.load_weights(name)
        self.critic.load_weights(name)

    # 학습된 모델을 저장함
    def save_model(self, name1, name2):
        self.actor.save_weights(name1)
        self.critic.save_weights(name2)


if __name__ == "__main__":
    # CartPole-v1의 경우 500 타임스텝까지 플레이가능
    env = gym.make('CartPole-v1')

    # 환경으로부터 상태와 행동의 크기를 가져옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = ACAgent(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # agent.load_model("./save_model/Cartpole-Actor.h5", "./save_model/Cartpole-Critic.h5")

        while not done:
            if agent.render:
                env.render()

            # 현재 상태에서 행동을 선택하고 한 스텝을 진행
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드를 끝나게 한 행동에 대해서 -100의 패널티를 줌
            reward = reward if not done or score == 499 else -100

            # <s, a, r, s'>을 replay memory에 저장
            agent.replay_memory(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습을 진행
            agent.train_replay()

            score += reward
            state = next_state

            if done:
                env.reset()

                # 각 에피소드마다 cartpole이 서있었던 타임스텝을 plot
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Cartpole_ActorCritc.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory))

                # 지난 10 에피소드의 평균이 490 이상이면 학습을 멈춤
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # 50 에피소드마다 학습 모델을 저장
        if e % 50 == 0:
            agent.save_model("./save_model/Cartpole_Actor.h5", "./save_model/Cartpole_Critic.h5")
