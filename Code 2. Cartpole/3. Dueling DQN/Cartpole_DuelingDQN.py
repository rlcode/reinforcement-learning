import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras import backend as k
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Lambda, merge, Input

EPISODES = 300


class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        # Cartpole이 학습하는 것을 보려면 "True"로 바꿀 것
        self.render = False

        # state와 action의 크기를 가져와서 모델을 생성하는데 사용함
        self.state_size = state_size
        self.action_size = action_size

        # Cartpole DQN 학습의 Hyper parameter 들
        # deque를 통해서 replay memory 생성
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 24
        self.train_start = 1000
        self.memory = deque(maxlen=2000)

        # 학습할 모델과 타겟 모델을 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 학습할 모델을 타겟 모델로 복사 --> 타겟 모델의 초기화(weight를 같게 해주고 시작해야 함)
        self.update_target_model()

    # Deep Neural Network를 통해서 Q Function을 근사
    # state가 입력, 각 행동에 대한 Q Value가 출력인 모델을 생성
    def build_model(self):
        input = Input(shape=(self.state_size,))
        x = Dense(32, input_shape=(self.state_size,), activation='relu', kernel_initializer='he_uniform')(input)
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(x)

        state_value = Dense(1, kernel_initializer='he_uniform')(x)
        state_value = Lambda(lambda s: k.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)

        action_advantage = Dense(self.action_size, kernel_initializer='he_uniform')(x)
        action_advantage = Lambda(lambda a: a[:, :] - k.mean(a[:, :], keepdims=True),
                                  output_shape=(self.action_size,))(action_advantage)

        q_value = merge([state_value, action_advantage], mode='sum')
        model = Model(input=input, output=q_value)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    # 일정한 시간 간격마다 타겟 모델을 현재 학습하고 있는 모델로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 행동의 선택은 현재 네트워크에 대해서 epsilon-greedy 정책을 사용
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # <s,a,r,s'>을 replay_memory에 저장함
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # replay memory에서 batch_size 만큼의 샘플들을 무작위로 뽑아서 학습
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            # 큐러닝에서와 같이 s'에서의 최대 Q Value를 가져옴. 단, 타겟 모델에서 가져옴
            if done:
                target[action] = reward
            else:
                # Double DQN의 핵심. 행동의 선택은 학습 모델로, 업데이트하는 값은 타겟 모델로
                a = np.argmax(self.model.predict(next_state)[0])
                target[action] = reward + self.discount_factor * \
                                          (self.target_model.predict(next_state)[0][a])

            update_input[i] = state
            update_target[i] = target

        # 학습할 정답인 타겟과 현재 자신의 값의 minibatch를 만들고 그것으로 한 번에 모델 업데이트
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

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
    # DQN 에이전트의 생성
    agent = DuelingDQNAgent(state_size, action_size)

    global_step = 0
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
            global_step += 1

            # 현재 상태에서 행동을 선택하고 한 스텝을 진행
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드를 끝나게 한 행동에 대해서 -100의 패널티를 줌
            reward = reward if not done or score == 500 else -100

            # <s, a, r, s'>을 replay memory에 저장
            agent.replay_memory(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습을 진행
            agent.train_replay()

            score += reward
            state = next_state

            if done:
                env.reset()
                agent.update_target_model()
                # 에피소드에 따른 score를 plot
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/Cartpole_Dueling_DQN.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

                # 지난 10 에피소드의 평균이 490 이상이면 학습을 멈춤
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # 50 에피소드마다 학습 모델을 저장
        if e % 50 == 0:
            agent.save_model("./save_model/Cartpole_DQN.h5")
