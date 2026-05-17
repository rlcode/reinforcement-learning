"""PPO agent for Atari (Breakout / Pong).

Schulman et al., 2017: "Proximal Policy Optimization Algorithms"
(arXiv:1707.06347).  Same clipped-surrogate + GAE objective as the
cartpole PPO, but with the Nature CNN as the shared trunk and the
DeepMind reward clipping that keeps the value function stable.

Single-env rollout for simplicity — real Atari PPO typically uses 8
parallel envs (CleanRL).  Bump TOTAL_FRAMES well past the default for
paper-quality results.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_env, parse_args, pick_device, quit_if_window_closed, run_test_loop


SAVE_PATH = "atari_ppo.pt"
TOTAL_FRAMES = 1_000_000
ROLLOUT_STEPS = 1024
EPOCHS = 4
MINIBATCH_SIZE = 256
CLIP_COEF = 0.1
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 2.5e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


# Nature CNN shared trunk + policy and value heads.
class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(4, 32, kernel_size=8, stride=4), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, kernel_size=4, stride=2), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, kernel_size=3, stride=1), 2 ** 0.5), nn.ReLU(),
            nn.Flatten(),
            _ortho(nn.Linear(64 * 7 * 7, 512), 2 ** 0.5), nn.ReLU(),
        )
        # gain=0.01 keeps the initial action distribution close to uniform.
        self.policy = _ortho(nn.Linear(512, n_actions), 0.01)
        self.value  = _ortho(nn.Linear(512, 1), 1.0)

    def forward(self, x):
        h = self.conv(x.float() / 255.0)
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
    args = parse_args()
    device = pick_device(args.device)
    env = make_env(args)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (4, 84, 84)

    model = ActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    def policy_action(obs):
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            logits, _ = model(t)
            return int(torch.distributions.Categorical(logits=logits).sample().item())

    if args.test:
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        run_test_loop(env, policy_action)

    print(f"device: {device},  env: {args.env},  actions: {n_actions}")

    n_updates = TOTAL_FRAMES // ROLLOUT_STEPS
    obs, _ = env.reset()
    ep_return = 0.0
    ep_returns = []

    for update in range(1, n_updates + 1):
        obs_buf  = np.zeros((ROLLOUT_STEPS, *obs_shape), dtype=np.uint8)
        act_buf  = np.zeros(ROLLOUT_STEPS, dtype=np.int64)
        logp_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        rew_buf  = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        done_buf = np.zeros(ROLLOUT_STEPS, dtype=np.float32)
        val_buf  = np.zeros(ROLLOUT_STEPS, dtype=np.float32)

        # --- Rollout ---
        for t in range(ROLLOUT_STEPS):
            quit_if_window_closed(env)
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
                logits, value = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            obs_buf[t]  = np.asarray(obs)
            act_buf[t]  = int(action.item())
            logp_buf[t] = float(logp.item())
            val_buf[t]  = float(value.item())

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            ep_return += reward
            rew_buf[t]  = float(np.sign(reward))  # DeepMind reward clipping
            done_buf[t] = float(done)

            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                next_obs, _ = env.reset()
            obs = next_obs

        # --- GAE ---
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            _, last_value = model(obs_t)
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value.item())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t      = torch.as_tensor(obs_buf, device=device)
        act_t      = torch.as_tensor(act_buf, device=device)
        old_logp_t = torch.as_tensor(logp_buf, device=device)
        adv_t      = torch.as_tensor(advantages, device=device)
        ret_t      = torch.as_tensor(returns, device=device)

        # --- PPO updates ---
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
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        if ep_returns:
            recent = ep_returns[-20:]
            print(f"update: {update:>4}  frames: {update * ROLLOUT_STEPS:>8}  "
                  f"recent_mean_return: {np.mean(recent):.1f}  episodes: {len(ep_returns)}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}")
