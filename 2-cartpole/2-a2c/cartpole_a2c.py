"""A2C (Advantage Actor-Critic) agent for CartPole-v1.

Mnih et al., 2016: "Asynchronous Methods for Deep Reinforcement Learning"
(A3C paper; A2C is the synchronous variant).

Two networks:
  - Actor pi_theta(a|s): policy over actions.
  - Critic V_w(s):       state-value baseline.

One-step TD advantage:

    A(s, a) = r + gamma * V_w(s') - V_w(s)        (= 0 on terminal s')

Updates (one-step, online — like a TD(0) actor-critic):

    Actor:   maximize  log pi_theta(a|s) * A(s, a)         (A is treated as constant)
    Critic:  minimize  ( V_w(s) - (r + gamma * V_w(s')) )^2

Subtracting V_w(s) is the variance-reduction baseline; using a learned V
(rather than the Monte-Carlo return) is what makes this *actor-critic*.
"""
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES = 1000


# Policy network: outputs logits over actions.
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# Value network: outputs a scalar V(s).
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
        # The critic typically uses a larger lr than the actor to keep the
        # baseline tracking the policy.
        self.actor_lr = 1e-3
        self.critic_lr = 5e-3

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    # Sample a ~ pi_theta(.|s).
    def get_action(self, state):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(state, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1).numpy()
        return int(np.random.choice(self.action_size, p=probs))

    # One-step online update.
    def train_model(self, state, action, reward, next_state, done):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32)

        value = self.critic(state_t)
        # TD target for the critic; treated as a constant for the gradient.
        with torch.no_grad():
            next_value = self.critic(next_state_t)
            target = torch.tensor(float(reward)) if done else reward + self.discount_factor * next_value
        # Advantage: A(s,a) = target - V(s).  Detach: the actor sees A as fixed.
        advantage = (target - value).detach()

        # Actor loss: -log pi(a|s) * A  (gradient ascent on log pi * A).
        logits = self.actor(state_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        actor_loss = -log_probs[action] * advantage

        # Critic loss: squared TD error.
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
            # Penalty on early termination (same shaping as DQN script).
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
