"""DQN agent for CartPole-v1.

Mnih et al., 2015: "Human-level control through deep reinforcement
learning" (Nature). Key ingredients vs. plain online Q-learning:

  1. Experience replay: store (s, a, r, s', done) and sample i.i.d.
     minibatches, breaking correlation between consecutive samples.
  2. Target network: a periodically-copied snapshot of the Q-network used
     to compute the TD target, which stabilizes bootstrapping.

Off-policy Q-learning target (with target network Q_phi):

    y = r + gamma * max_{a'} Q_phi(s', a')      if not done
    y = r                                       if done

Loss (per minibatch sample):

    L(theta) = ( Q_theta(s)[a] - y )^2
"""
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_env, parse_args, run_test_loop

EPISODES = 300
SAVE_PATH = "cartpole_dqn.pt"


# Approximator for Q(s, .). He-uniform init is friendly to ReLU.
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

        # Hyperparameters.
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        # Wait until the replay buffer has enough samples before training.
        self.train_start = 1000
        # Replay memory: a sliding window of recent transitions.
        self.memory = deque(maxlen=2000)

        # Online network (trained) and target network (slow copy for bootstrapping).
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    # Hard update: target <- online. Called once per episode.
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Epsilon-greedy over Q_theta(s, .).
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q = self.model(torch.as_tensor(state, dtype=torch.float32))
        return int(torch.argmax(q).item())

    # Store transition <s, a, r, s', done> and decay epsilon.
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # One SGD step on a uniformly-sampled minibatch from the replay buffer.
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

        # Q_theta(s, a)
        q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # y = r + gamma * max_a' Q_phi(s', a')   (zeroed out on terminal s')
        with torch.no_grad():
            q_next = self.target_model(next_states).max(dim=1).values
            target = rewards + (1.0 - dones) * self.discount_factor * q_next

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    env = make_env(args)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    if args.test:
        agent.model.load_state_dict(torch.load(SAVE_PATH))
        agent.epsilon = 0.0  # fully greedy
        run_test_loop(env, agent.get_action)

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
            # Reward shaping: heavy penalty for early termination encourages
            # balancing rather than treating any +1 as success.
            shaped_reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, shaped_reward, next_state, done)
            # Train at every environment step.
            agent.train_model()
            score += shaped_reward
            state = next_state

            if done:
                # Update target network once per episode.
                agent.update_target_model()
                # Undo the shaping penalty for the displayed score.
                score = score if score == 500 else score + 100
                scores.append(score)
                print(f"episode: {e}  score: {score}  memory: {len(agent.memory)}  epsilon: {agent.epsilon:.4f}")

                # Early stop when consistently near max episode length.
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model.state_dict(), SAVE_PATH)
                    print(f"Saved trained model to {SAVE_PATH}")
                    sys.exit()

    torch.save(agent.model.state_dict(), SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}")
