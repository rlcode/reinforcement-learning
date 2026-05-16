import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import Env

EPISODES = 1000


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, action_size),
        )

    def forward(self, x):
        return self.net(x)


class DeepSARSAgent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

        self.model = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(torch.as_tensor(state, dtype=torch.float32))
        return int(torch.argmax(q_values).item())

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state_t = torch.as_tensor(state, dtype=torch.float32)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32)

        q_pred = self.model(state_t)[action]
        with torch.no_grad():
            if done:
                target = torch.tensor(float(reward))
            else:
                next_q = self.model(next_state_t)[next_action]
                target = reward + self.discount_factor * next_q

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    env = Env()
    agent = DeepSARSAgent()
    global_step = 0

    for e in range(EPISODES):
        done = False
        score = 0
        state = np.array(env.reset(), dtype=np.float32)

        while not done:
            global_step += 1
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)
            state = next_state
            score += reward

            if done:
                print(f"episode: {e}  score: {score}  steps: {global_step}  epsilon: {agent.epsilon:.4f}")
