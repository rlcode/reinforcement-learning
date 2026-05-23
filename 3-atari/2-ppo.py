"""PPO agent for Atari (Breakout / Pong).

Schulman et al., 2017: "Proximal Policy Optimization Algorithms"
(arXiv:1707.06347).  Same clipped-surrogate + GAE objective as the
cartpole PPO, but with the Nature CNN as the shared trunk and the
DeepMind reward clipping that keeps the value function stable.

Rollout uses 8 parallel envs via SyncVectorEnv (CleanRL convention).
Bump TOTAL_FRAMES well past the default for paper-quality results.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_env, make_vec_env, parse_args, pick_device, run_test_loop


SAVE_PATH = "atari_ppo.pt"
TOTAL_FRAMES = 10_000_000
N_ENVS = 8
ROLLOUT_STEPS = 128            # batch = N_ENVS * ROLLOUT_STEPS = 1024
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

    if args.test:
        env = make_env(args)
        n_actions = env.action_space.n
        model = ActorCritic(n_actions).to(device)
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        def policy_action(obs):
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
                logits, _ = model(t)
                return int(torch.distributions.Categorical(logits=logits).sample().item())
        run_test_loop(env, policy_action)

    envs = make_vec_env(args, N_ENVS)
    n_actions = envs.single_action_space.n
    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)

    model = ActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    if args.wandb:
        import wandb
        wandb.init(project="rl-atari-ppo", config={
            "env": args.env, "n_envs": N_ENVS, "rollout_steps": ROLLOUT_STEPS,
            "total_frames": TOTAL_FRAMES, "epochs": EPOCHS,
            "minibatch_size": MINIBATCH_SIZE, "clip_coef": CLIP_COEF,
            "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "lr": LR,
            "value_coef": VALUE_COEF, "entropy_coef": ENTROPY_COEF,
        })

    print(f"device: {device},  env: {args.env},  actions: {n_actions},  n_envs: {N_ENVS}")

    batch_size = ROLLOUT_STEPS * N_ENVS
    frames_per_update = batch_size
    n_updates = TOTAL_FRAMES // frames_per_update
    obs, _ = envs.reset()
    ep_returns_per_env = np.zeros(N_ENVS, dtype=np.float32)    # per-life (resets every life loss)
    game_returns_per_env = np.zeros(N_ENVS, dtype=np.float32)  # per-game (resets only on real game-over)
    ep_returns = []
    game_returns = []

    for update in range(1, n_updates + 1):
        # Linear LR anneal from LR -> 0 over the run (CleanRL convention).
        lr_now = LR * (1.0 - (update - 1) / n_updates)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        obs_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS, *obs_shape), dtype=np.uint8)
        act_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.int64)
        logp_buf = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        rew_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        done_buf = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        val_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)

        # --- Rollout ---
        for t in range(ROLLOUT_STEPS):
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs), device=device)
                logits, value = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            obs_buf[t]  = np.asarray(obs)
            act_buf[t]  = action.cpu().numpy()
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t]  = value.cpu().numpy()

            next_obs, reward, terminated, truncated, info = envs.step(act_buf[t])
            done = np.logical_or(terminated, truncated)
            ep_returns_per_env += reward
            game_returns_per_env += reward
            rew_buf[t]  = np.sign(reward).astype(np.float32)  # DeepMind reward clipping
            done_buf[t] = done.astype(np.float32)

            # LifeLossTerminalEnv tags each step's info with game_over (True only on real game-over).
            game_over = info.get("game_over", done)
            for i in range(N_ENVS):
                if done[i]:
                    ep_returns.append(float(ep_returns_per_env[i]))
                    ep_returns_per_env[i] = 0.0
                    if bool(game_over[i]):
                        game_returns.append(float(game_returns_per_env[i]))
                        game_returns_per_env[i] = 0.0
            obs = next_obs

        # --- GAE ---
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), device=device)
            _, last_value = model(obs_t)
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value.cpu().numpy())

        # Flatten (T, N_ENVS, ...) -> (T*N_ENVS, ...)
        obs_t      = torch.as_tensor(obs_buf.reshape(batch_size, *obs_shape), device=device)
        act_t      = torch.as_tensor(act_buf.reshape(batch_size), device=device)
        old_logp_t = torch.as_tensor(logp_buf.reshape(batch_size), device=device)
        old_val_t  = torch.as_tensor(val_buf.reshape(batch_size), device=device)
        adv_t      = torch.as_tensor(advantages.reshape(batch_size), device=device)
        ret_t      = torch.as_tensor(returns.reshape(batch_size), device=device)

        # --- PPO updates ---
        idx = np.arange(batch_size)
        pl_sum = vl_sum = ent_sum = 0.0
        n_mb = 0
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                logits, values = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                # Advantage normalization per minibatch (CleanRL convention).
                mb_adv = adv_t[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = (new_logp - old_logp_t[mb]).exp()
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss with clipping around the old value prediction.
                v_clipped = old_val_t[mb] + torch.clamp(
                    values - old_val_t[mb], -CLIP_COEF, CLIP_COEF)
                vl_unclipped = (values - ret_t[mb]).pow(2)
                vl_clipped   = (v_clipped - ret_t[mb]).pow(2)
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                pl_sum += policy_loss.item()
                vl_sum += value_loss.item()
                ent_sum += entropy.item()
                n_mb += 1

        global_step = update * frames_per_update
        if ep_returns:
            life_mean = float(np.mean(ep_returns[-20:]))
            game_mean = float(np.mean(game_returns[-20:])) if game_returns else 0.0
            print(f"update: {update:>4}  frames: {global_step:>8}  "
                  f"per_life: {life_mean:.1f}  per_game: {game_mean:.1f}  "
                  f"lives: {len(ep_returns)}  games: {len(game_returns)}")
        if args.wandb:
            log = {
                "global_step": global_step,
                "policy_loss": pl_sum / n_mb,
                "value_loss": vl_sum / n_mb,
                "entropy": ent_sum / n_mb,
                "lr": lr_now,
            }
            if ep_returns:
                log["recent_mean_return"] = float(np.mean(ep_returns[-20:]))
            if game_returns:
                log["recent_mean_game_return"] = float(np.mean(game_returns[-20:]))
            wandb.log(log, step=global_step)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}")
