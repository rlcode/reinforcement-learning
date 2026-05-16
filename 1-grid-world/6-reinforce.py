"""REINFORCE (Monte-Carlo policy gradient) agent for the GridWorld.

Williams, 1992: "Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning".

Policy gradient theorem:

    grad_theta J(theta) = E_pi [ grad_theta log pi_theta(a|s) * G_t ]

where G_t = sum_{k>=t} gamma^(k-t) * r_k is the return from step t.

We use the per-episode Monte-Carlo estimator: collect a full trajectory,
compute discounted returns G_t, then ascend the gradient. The returns are
standardized (zero-mean, unit-variance) as a simple variance-reduction
trick (acts like a constant baseline).

Implementation note: we maximize expected return, i.e. minimize the
negative log-likelihood weighted by G_t.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import DynamicEnv

EPISODES = 2500


# Policy network: state -> logits over actions.
# Softmax is applied where we need probabilities (sampling / log-prob).
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReinforceAgent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 1e-3

        self.model = PolicyNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Per-episode trajectory buffer.
        self.states, self.actions, self.rewards = [], [], []

    # Sample a ~ pi_theta(.|s).
    def get_action(self, state):
        with torch.no_grad():
            logits = self.model(torch.as_tensor(state, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1).numpy()
        return int(np.random.choice(self.action_size, p=probs))

    # G_t = r_t + gamma * G_{t+1}, computed backwards from the episode end.
    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = running * self.discount_factor + rewards[t]
            discounted[t] = running
        return discounted

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # Single gradient step using the whole episode.
    def train_model(self):
        returns = self.discount_rewards(np.array(self.rewards, dtype=np.float32))
        # Variance-reduction baseline (standardization).
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.as_tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.as_tensor(self.actions, dtype=torch.long)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        # log pi_theta(a_t | s_t) for each step in the trajectory.
        logits = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Negative log-likelihood weighted by return -> minimize == ascend policy gradient.
        loss = -(chosen * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # REINFORCE uses a per-step -0.1 penalty to encourage shorter paths.
    env = DynamicEnv(title="REINFORCE", step_penalty=0.1)
    agent = ReinforceAgent()
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

            agent.append_sample(state, action, reward)
            score += reward
            state = next_state

            if done:
                # REINFORCE updates once per episode (Monte-Carlo).
                agent.train_model()
                print(f"episode: {e}  score: {round(score, 2)}  steps: {global_step}")
