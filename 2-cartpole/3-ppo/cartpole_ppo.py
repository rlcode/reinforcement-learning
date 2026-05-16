"""Minimal PPO for CartPole-v1 in the spirit of CleanRL's single-file style."""
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES = 1000
ROLLOUT_STEPS = 256
EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_COEF = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy = nn.Linear(64, action_size)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)


def compute_gae(rewards, values, dones, last_value):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_v * next_nonterminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = ActorCritic(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    ep_return = 0.0
    ep_returns = []

    for episode in range(EPISODES):
        obs_buf = np.zeros((ROLLOUT_STEPS, state_size), dtype=np.float32)
        act_buf = np.zeros(ROLLOUT_STEPS, dtype=np.int64)
        logp_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        rew_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        done_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        val_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)

        for t in range(ROLLOUT_STEPS):
            with torch.no_grad():
                logits, value = model(torch.as_tensor(state))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            obs_buf[t] = state
            act_buf[t] = action.item()
            logp_buf[t] = logp.item()
            val_buf[t] = value.item()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rew_buf[t] = reward
            done_buf[t] = float(done)
            ep_return += reward

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                next_state, _ = env.reset()
            state = np.array(next_state, dtype=np.float32)

        with torch.no_grad():
            _, last_value = model(torch.as_tensor(state))
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value.item())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs_buf)
        act_t = torch.as_tensor(act_buf)
        old_logp_t = torch.as_tensor(logp_buf)
        adv_t = torch.as_tensor(advantages)
        ret_t = torch.as_tensor(returns)

        idx = np.arange(ROLLOUT_STEPS)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, ROLLOUT_STEPS, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                logits, values = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logp - old_logp_t[mb]).exp()
                unclipped = ratio * adv_t[mb]
                clipped = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * adv_t[mb]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (values - ret_t[mb]).pow(2).mean()
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if ep_returns:
            recent = ep_returns[-10:]
            print(f"update: {episode}  recent_mean_return: {np.mean(recent):.1f}  episodes: {len(ep_returns)}")
            if len(recent) >= 10 and np.mean(recent) > 490:
                sys.exit()
