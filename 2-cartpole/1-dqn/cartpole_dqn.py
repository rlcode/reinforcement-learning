import random
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES = 300


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=2000)

        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q = self.model(torch.as_tensor(state, dtype=torch.float32))
        return int(torch.argmax(q).item())

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_model(next_states).max(dim=1).values
            target = rewards + (1.0 - dones) * self.discount_factor * q_next

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    scores = []

    for e in range(EPISODES):
        done = False
        score = 0
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)
            shaped_reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, shaped_reward, next_state, done)
            agent.train_model()
            score += shaped_reward
            state = next_state

            if done:
                agent.update_target_model()
                score = score if score == 500 else score + 100
                scores.append(score)
                print(f"episode: {e}  score: {score}  memory: {len(agent.memory)}  epsilon: {agent.epsilon:.4f}")

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
