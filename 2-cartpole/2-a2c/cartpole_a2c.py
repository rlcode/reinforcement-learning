import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES = 1000


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.actor_lr = 1e-3
        self.critic_lr = 5e-3

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def get_action(self, state):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(state, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1).numpy()
        return int(np.random.choice(self.action_size, p=probs))

    def train_model(self, state, action, reward, next_state, done):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32)

        value = self.critic(state_t)
        with torch.no_grad():
            next_value = self.critic(next_state_t)
            target = torch.tensor(float(reward)) if done else reward + self.discount_factor * next_value
        advantage = (target - value).detach()

        logits = self.actor(state_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        actor_loss = -log_probs[action] * advantage

        critic_loss = (value - target).pow(2)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)
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

            agent.train_model(state, action, shaped_reward, next_state, done)
            score += shaped_reward
            state = next_state

            if done:
                score = score if score == 500.0 else score + 100
                scores.append(score)
                print(f"episode: {e}  score: {score}")
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
