import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from collections import deque
from pygame.constants import K_a, K_d

def process_state(state):
    return np.array([state.values()])

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=200000)
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9984
        self.learning_rate = 1e-5
        self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=4, activation='tanh', init='he_uniform'))
        model.add(Dense(100, activation='tanh', init='he_uniform'))
        model.add(Dense(3, activation='linear', init='he_uniform'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model



    def remember(self, state, action, reward, next_state, done):  #메모리 저장
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([K_a, None, K_d])
        act_values = self.model.predict(state)
        # print (act_values)
        return [K_a, None, K_d][np.argmax(act_values[0])]

    def replay(self, batch_size):
        if len(self.memory) < 120000:   #메모리 사이즈가 120000 이하면 학습 안함
            return
        batchs = np.random.choice(len(self.memory), batch_size, replace=False)  #배치사이즈만큼 메모리 랜덤하게 가져오기
        states, targets = [], []
        for i in batchs:
            state, action, reward, next_state, done = self.memory[i]
            #if not done:
            target = reward + self.gamma * \
                              np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, nb_epoch=1, verbose=0)  # 학습하기
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):   # 학습된 네트워크 로드
        self.model.load_weights(name)

    def save(self, name):   # 네트워크 저장
        self.model.save_weights(name)