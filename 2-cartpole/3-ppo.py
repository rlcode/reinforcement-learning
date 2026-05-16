"""PPO (Proximal Policy Optimization) agent for CartPole-v1.

Schulman et al., 2017: "Proximal Policy Optimization Algorithms"
(arXiv:1707.06347).  Also uses GAE from Schulman et al., 2016:
"High-Dimensional Continuous Control Using Generalized Advantage
Estimation" (arXiv:1506.02438).

PPO is an on-policy actor-critic method. Define the probability ratio:

    r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)

Clipped surrogate objective (the heart of PPO):

    L^CLIP(theta) = E_t [ min( r_t(theta) * A_t,
                               clip(r_t(theta), 1 - eps, 1 + eps) * A_t ) ]

By clipping the ratio we discourage updates that move pi too far from
pi_old in a single step — this is what lets us reuse a batch of data
for several gradient epochs while staying near the trust region.

Generalized Advantage Estimation (GAE-lambda):

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}

Total loss combines clipped policy loss, value MSE, and an entropy bonus:

    L = L^CLIP - c_v * MSE(V, returns) + c_e * H[pi]
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_env, parse_args, quit_if_window_closed, run_test_loop

EPISODES = 1500
SAVE_PATH = "cartpole_ppo.pt"
# Steps collected per update; PPO is batch-based, not single-step like A2C.
# 256 is too small for a single env on CartPole — GAE gets noisy and PPO
# oscillates.  1024 (with 4 epochs / 64 minibatches) is closer to the
# CleanRL single-env reference and gives much steadier learning.
ROLLOUT_STEPS = 1024
# Number of times we sweep over the collected batch each update.
EPOCHS = 4
MINIBATCH_SIZE = 64
# Clip range epsilon from the PPO paper; 0.2 is the canonical value.
CLIP_COEF = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
# Value-loss weight and entropy bonus weight.
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


def _ortho(layer, gain):
    """Orthogonal init — a standard PPO stability trick (CleanRL-style)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


# Shared-trunk actor-critic: two-layer MLP with tanh, then policy and value heads.
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # gain = sqrt(2) for the tanh trunk, 0.01 for the policy head
        # (keeps initial action distribution close to uniform), 1 for the
        # value head.  These are the standard PPO-paper / CleanRL choices.
        self.shared = nn.Sequential(
            _ortho(nn.Linear(state_size, 64), gain=2 ** 0.5),
            nn.Tanh(),
            _ortho(nn.Linear(64, 64), gain=2 ** 0.5),
            nn.Tanh(),
        )
        self.policy = _ortho(nn.Linear(64, action_size), gain=0.01)
        self.value = _ortho(nn.Linear(64, 1), gain=1.0)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)


# GAE-lambda: backward recursion over the collected rollout.
# `dones` marks terminal transitions; the recursion is reset there.
def compute_gae(rewards, values, dones, last_value):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + GAMMA * next_v * next_nonterminal - values[t]
        # A_t = delta_t + gamma * lambda * A_{t+1}
        gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * gae
        advantages[t] = gae
    # Returns used as the value target: R_t = A_t + V(s_t).
    returns = advantages + values
    return advantages, returns


if __name__ == "__main__":
    args = parse_args()
    env = make_env(args)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = ActorCritic(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if args.test:
        model.load_state_dict(torch.load(SAVE_PATH))

        def pick(state):
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(state))
                return int(torch.distributions.Categorical(logits=logits).sample().item())

        run_test_loop(env, pick)

    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    ep_return = 0.0
    ep_returns = []

    for episode in range(EPISODES):
        # --- 1. Roll out the current policy for ROLLOUT_STEPS. ---
        obs_buf = np.zeros((ROLLOUT_STEPS, state_size), dtype=np.float32)
        act_buf = np.zeros(ROLLOUT_STEPS, dtype=np.int64)
        logp_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        rew_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        done_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        val_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)

        for t in range(ROLLOUT_STEPS):
            quit_if_window_closed(env)
            with torch.no_grad():
                logits, value = model(torch.as_tensor(state))
                # Categorical handles softmax + sampling + log_prob cleanly.
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            obs_buf[t] = state
            act_buf[t] = action.item()
            # Stash log pi_theta_old(a_t | s_t) for the ratio computation later.
            logp_buf[t] = logp.item()
            val_buf[t] = value.item()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            ep_return += reward  # raw episode length (for reporting)
            # Reward shaping (matches DQN / A2C / rlcode-kr-v2): +0.1 per
            # surviving step, -1 on the failure step.  Without this PPO
            # gets a very weak signal on CartPole and oscillates.
            rew_buf[t] = 0.1 if not done or ep_return == 500 else -1
            done_buf[t] = float(done)

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                next_state, _ = env.reset()
            state = np.array(next_state, dtype=np.float32)

        # --- 2. Compute advantages and returns via GAE. ---
        # Bootstrap with V(s_T) at the rollout boundary (not necessarily terminal).
        with torch.no_grad():
            _, last_value = model(torch.as_tensor(state))
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value.item())
        # Per-batch advantage normalization (standard PPO trick).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs_buf)
        act_t = torch.as_tensor(act_buf)
        old_logp_t = torch.as_tensor(logp_buf)
        adv_t = torch.as_tensor(advantages)
        ret_t = torch.as_tensor(returns)

        # --- 3. Multiple epochs of minibatch SGD on the clipped surrogate. ---
        idx = np.arange(ROLLOUT_STEPS)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, ROLLOUT_STEPS, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                logits, values = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                # ratio = pi_new / pi_old = exp(log pi_new - log pi_old)
                ratio = (new_logp - old_logp_t[mb]).exp()
                unclipped = ratio * adv_t[mb]
                clipped = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * adv_t[mb]
                # PPO objective is the *min* of clipped and unclipped — pessimistic
                # bound that ignores improvements outside the trust region.
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (values - ret_t[mb]).pow(2).mean()
                # Entropy bonus encourages exploration.
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                # Global grad clipping is a standard stabilizer in PPO.
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if ep_returns:
            recent = ep_returns[-10:]
            print(f"update: {episode}  recent_mean_return: {np.mean(recent):.1f}  episodes: {len(ep_returns)}")
            if len(recent) >= 10 and np.mean(recent) > 490:
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"Saved trained model to {SAVE_PATH}")
                sys.exit()

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}")
