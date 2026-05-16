"""DQN agent for Atari (Breakout / Pong).

Mnih et al., 2015: "Human-level control through deep reinforcement
learning" (Nature).  Same algorithm as the cartpole DQN but with the
Nature CNN backbone, a much bigger replay buffer, reward clipping, and
a slower target-network refresh interval.

Real Atari runs take tens of millions of frames to converge; the
defaults below are tuned to be *runnable* on a laptop rather than to
hit DeepMind-paper scores.  Bump TOTAL_FRAMES and BUFFER_CAPACITY for
serious training.
"""
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_env, parse_args, pick_device, quit_if_window_closed, run_test_loop


SAVE_PATH = "atari_dqn.pt"
TOTAL_FRAMES = 1_000_000        # bump to ~10M for paper-quality results
BUFFER_CAPACITY = 100_000       # bump to 1M with enough RAM
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
LEARN_START = 10_000            # frames of pure exploration before training begins
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1_000     # in training steps, not env steps
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRAMES = 250_000  # linear decay from start to end over this many frames


# Standard Nature CNN.
class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # Inputs are uint8 in [0, 255]; normalize on the GPU to save bus bandwidth.
        return self.fc(self.conv(x.float() / 255.0))


class ReplayBuffer:
    """Uint8 replay buffer — far more memory-efficient than storing floats."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def epsilon(frame):
    """Linear schedule from EPSILON_START to EPSILON_END over EPSILON_DECAY_FRAMES."""
    frac = min(frame / EPSILON_DECAY_FRAMES, 1.0)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


if __name__ == "__main__":
    args = parse_args()
    device = pick_device(args.device)
    env = make_env(args)
    n_actions = env.action_space.n

    online = QNetwork(n_actions).to(device)
    target = QNetwork(n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()  # Huber loss — standard for DQN

    def greedy_action(obs):
        """Used by --test and during exploitation steps."""
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            return int(online(t).argmax(dim=1).item())

    if args.test:
        online.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        run_test_loop(env, greedy_action)

    print(f"device: {device},  env: {args.env},  actions: {n_actions}")

    buffer = ReplayBuffer(BUFFER_CAPACITY, env.observation_space.shape)
    obs, _ = env.reset()
    ep_return = 0.0
    recent_returns = deque(maxlen=20)
    train_step = 0

    for frame in range(1, TOTAL_FRAMES + 1):
        quit_if_window_closed(env)

        # Epsilon-greedy action.
        if random.random() < epsilon(frame):
            action = env.action_space.sample()
        else:
            action = greedy_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Reward clipping (DeepMind standard) — keeps Q-values from blowing up
        # when one game has rewards in tens and another in hundreds.
        clipped = np.sign(reward)
        buffer.push(np.asarray(obs), action, clipped, np.asarray(next_obs), done)

        ep_return += reward
        obs = next_obs
        if done:
            recent_returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()

        # Training.
        if frame > LEARN_START and frame % TRAIN_EVERY == 0:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE, device)
            q_pred = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target(next_states).max(dim=1).values
                y = rewards + (1.0 - dones) * GAMMA * q_next
            loss = loss_fn(q_pred, y)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — DeepMind uses global norm 10.
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()

            train_step += 1
            if train_step % TARGET_UPDATE_EVERY == 0:
                target.load_state_dict(online.state_dict())

        # Logging.
        if frame % 10_000 == 0:
            mean = np.mean(recent_returns) if recent_returns else 0.0
            print(f"frame: {frame:>8}  eps: {epsilon(frame):.3f}  "
                  f"recent_mean_return: {mean:.1f}  buffer: {buffer.size}")

    torch.save(online.state_dict(), SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}")
