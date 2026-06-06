"""PPO + Count-Based Exploration (SimHash) for hard-exploration Atari.

The algorithmic contrast to 1-ppo-rnd.py. RND on Montezuma gave 0 across
three configurations (see W&B runs dulcet/prime/usual) because the
intrinsic signal stayed uniform across all 128 envs — predictor's "novelty"
caught the agent's micro-variation inside Room 1 but had no concept of
"never seen this kind of frame before". Count-based methods fix this by
*explicitly remembering* visited states.

Tang et al., 2017: "#Exploration: A Study of Count-Based Exploration for
Deep RL" (arXiv:1611.04717). Pixel-similar frames hash to the same bit
string via a fixed random projection (locality-sensitive hashing). For
each state s we track N(s) = number of times we've hashed there, and give
the policy an intrinsic bonus:

    r_int(s) = 1 / sqrt(N(s) + 1)

A brand-new state pays 1.0; a state seen 100 times pays 0.1. Every
discovered cell stays discovered forever — there is no "predictor catching
up" failure mode.

Reuses the dual-value-head + dual-GAE + intrinsic-reward normalization
scaffolding from 1-ppo-rnd.py. Drops obs RMS, RND target/predictor,
predictor loss, and the random-agent warmup (counts work from t=0).
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_vec_env, parse_args, pick_device


SAVE_PATH = "atari_count_based.pt"
TOTAL_FRAMES = 10_000_000
N_ENVS = 64                    # envpool: count-based less memory-hungry than RND, but stay safe on 8 GB
ROLLOUT_STEPS = 128            # batch = 8192
EPOCHS = 4
MINIBATCH_SIZE = 2048          # 4 minibatches per epoch
CLIP_COEF = 0.1
GAMMA_EXT = 0.999              # sparse-reward games need long horizons
GAMMA_INT = 0.99
GAE_LAMBDA = 0.95
LR = 1e-4
EXT_COEF = 2.0
INT_COEF = 1.0
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01            # same lesson as RND: keep exploration alive at 10M scale
MAX_GRAD_NORM = 0.5

HASH_DIM = 64                  # smaller hash = coarser lumping. Raw 84x84 + 256 bits made
                               # 87% of frames hash to new cells (every pixel jiggle = new state).
DOWNSAMPLE_TO = 8              # avg-pool 84x84 -> 8x8 before hashing — keep position/room
                               # signal, drop pixel-level noise.
FRAME_SIZE = DOWNSAMPLE_TO * DOWNSAMPLE_TO


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


# Same Nature CNN trunk + two value heads as 1-ppo-rnd.py.
class ActorCriticCount(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(4, 32, kernel_size=8, stride=4), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, kernel_size=4, stride=2), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, kernel_size=3, stride=1), 2 ** 0.5), nn.ReLU(),
            nn.Flatten(),
            _ortho(nn.Linear(64 * 7 * 7, 512), 2 ** 0.5), nn.ReLU(),
        )
        self.policy = _ortho(nn.Linear(512, n_actions), 0.01)
        self.value_ext = _ortho(nn.Linear(512, 1), 1.0)
        self.value_int = _ortho(nn.Linear(512, 1), 1.0)

    def forward(self, x):
        h = self.conv(x.float() / 255.0)
        return self.policy(h), self.value_ext(h).squeeze(-1), self.value_int(h).squeeze(-1)


class CountHasher:
    """Locality-sensitive hash: pixel-similar frames -> same bit string.

    Two-stage compression: each 84x84 grayscale frame is first average-pooled
    to DOWNSAMPLE_TO x DOWNSAMPLE_TO (drops sub-tile pixel noise, keeps room
    layout and agent position), then a fixed random projection maps that to a
    HASH_DIM-bit signature. Sign of the projection is the hash key.

    Each visit increments the count for that key and returns 1/sqrt(count) as
    the intrinsic reward. No neural net, no warmup — discovery is permanent
    the moment a new hash key appears."""
    def __init__(self, hash_dim=64, downsample_to=8, src_size=84, seed=0):
        rng = np.random.default_rng(seed)
        frame_size = downsample_to * downsample_to
        self.A = (rng.standard_normal((frame_size, hash_dim)).astype(np.float32)
                  / np.sqrt(frame_size))
        self.counts = defaultdict(int)
        self.src_size = src_size
        self.down = downsample_to
        # Symmetric crop so src_size becomes divisible by down (e.g. 84 -> 80 for down=8).
        cropped = (src_size // downsample_to) * downsample_to
        self.crop = (src_size - cropped) // 2
        self.block = cropped // downsample_to

    def downsample(self, frames):
        """(n, 84, 84) uint8 -> (n, down, down) float32 (mean over blocks)."""
        c = self.crop
        cropped = frames[:, c:c + self.down * self.block, c:c + self.down * self.block]
        return (cropped.astype(np.float32)
                .reshape(len(frames), self.down, self.block, self.down, self.block)
                .mean(axis=(2, 4)))

    def visit(self, frames):
        """frames: (n, 84, 84) uint8 — returns (n,) float32 intrinsic rewards
        AND increments the corresponding counts."""
        pooled = self.downsample(frames)
        flat = pooled.reshape(len(frames), -1)
        bits = np.sign(flat @ self.A).astype(np.int8)
        rewards = np.empty(len(frames), dtype=np.float32)
        for i, row in enumerate(bits):
            key = row.tobytes()
            self.counts[key] += 1
            rewards[i] = 1.0 / np.sqrt(self.counts[key])
        return rewards

    def num_cells(self):
        return len(self.counts)


class RunningMeanStd:
    """Welford / Chan parallel algorithm. Used to scale the intrinsic
    return so the value head sees a stable target."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        bm = batch.mean(axis=0)
        bv = batch.var(axis=0)
        bc = batch.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + delta ** 2 * self.count * bc / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot


def compute_gae(rewards, values, nonterminals, last_value, gamma, lam):
    """Pass nonterminals=1-dones for episodic (extrinsic); all-ones for
    non-episodic intrinsic so curiosity chains across deaths."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminals[t] - values[t]
        gae = delta + gamma * lam * nonterminals[t] * gae
        advantages[t] = gae
    return advantages, advantages + values


if __name__ == "__main__":
    args = parse_args()
    device = pick_device(args.device)
    envs = make_vec_env(args, N_ENVS)
    n_actions = envs.action_space.n
    obs_shape = envs.observation_space.shape  # (4, 84, 84)

    model = ActorCriticCount(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    hasher = CountHasher(hash_dim=HASH_DIM, downsample_to=DOWNSAMPLE_TO, src_size=84)
    int_ret_rms = RunningMeanStd(shape=())

    if args.wandb:
        import wandb
        wandb.init(project="rl-atari-hard-count-based", config={
            "env": args.env, "n_envs": N_ENVS, "rollout_steps": ROLLOUT_STEPS,
            "total_frames": TOTAL_FRAMES, "epochs": EPOCHS,
            "minibatch_size": MINIBATCH_SIZE, "clip_coef": CLIP_COEF,
            "gamma_ext": GAMMA_EXT, "gamma_int": GAMMA_INT, "gae_lambda": GAE_LAMBDA,
            "lr": LR, "ext_coef": EXT_COEF, "int_coef": INT_COEF,
            "value_coef": VALUE_COEF, "entropy_coef": ENTROPY_COEF,
            "hash_dim": HASH_DIM,
        })

    print(f"device: {device},  env: {args.env},  actions: {n_actions},  n_envs: {N_ENVS}")

    batch_size = ROLLOUT_STEPS * N_ENVS
    n_updates = TOTAL_FRAMES // batch_size
    obs, _ = envs.reset()
    ep_returns_per_env = np.zeros(N_ENVS, dtype=np.float32)
    ep_returns = []
    int_filter = np.zeros(N_ENVS, dtype=np.float64)

    for update in range(1, n_updates + 1):
        lr_now = LR * (1.0 - (update - 1) / n_updates)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        obs_buf      = np.zeros((ROLLOUT_STEPS, N_ENVS, *obs_shape), dtype=np.uint8)
        act_buf      = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.int64)
        logp_buf     = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        rew_ext_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        rew_int_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        val_ext_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        val_int_buf  = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
        done_buf     = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)

        # --- Rollout ---
        for t in range(ROLLOUT_STEPS):
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs), device=device)
                logits, v_ext, v_int = model(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            obs_buf[t]     = np.asarray(obs)
            act_buf[t]     = action.cpu().numpy()
            logp_buf[t]    = logp.cpu().numpy()
            val_ext_buf[t] = v_ext.cpu().numpy()
            val_int_buf[t] = v_int.cpu().numpy()

            next_obs, reward, terminated, truncated, _ = envs.step(act_buf[t].astype(np.int32))
            done = np.logical_or(terminated, truncated)
            ep_returns_per_env += reward
            rew_ext_buf[t] = np.sign(reward).astype(np.float32)
            done_buf[t]    = done.astype(np.float32)

            # Count-based intrinsic reward on the newest frame of next_obs.
            next_last = np.asarray(next_obs)[:, -1, :, :]
            rew_int_buf[t] = hasher.visit(next_last)

            for i in range(N_ENVS):
                if done[i]:
                    ep_returns.append(float(ep_returns_per_env[i]))
                    ep_returns_per_env[i] = 0.0
            obs = next_obs

        # --- Intrinsic reward normalization (running std of discounted intrinsic returns) ---
        for t in range(ROLLOUT_STEPS):
            int_filter = int_filter * GAMMA_INT + rew_int_buf[t]
            int_ret_rms.update(int_filter.copy())
        rew_int_buf = rew_int_buf / np.sqrt(int_ret_rms.var + 1e-8)

        # --- Dual GAE ---
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), device=device)
            _, last_v_ext, last_v_int = model(obs_t)
        adv_ext, ret_ext = compute_gae(
            rew_ext_buf, val_ext_buf, 1.0 - done_buf,
            last_v_ext.cpu().numpy(), GAMMA_EXT, GAE_LAMBDA,
        )
        adv_int, ret_int = compute_gae(
            rew_int_buf, val_int_buf, np.ones_like(done_buf),
            last_v_int.cpu().numpy(), GAMMA_INT, GAE_LAMBDA,
        )
        adv_combined = EXT_COEF * adv_ext + INT_COEF * adv_int

        obs_t      = torch.as_tensor(obs_buf.reshape(batch_size, *obs_shape), device=device)
        act_t      = torch.as_tensor(act_buf.reshape(batch_size), device=device)
        old_logp_t = torch.as_tensor(logp_buf.reshape(batch_size), device=device)
        adv_t      = torch.as_tensor(adv_combined.reshape(batch_size), device=device)
        ret_ext_t  = torch.as_tensor(ret_ext.reshape(batch_size), device=device)
        ret_int_t  = torch.as_tensor(ret_int.reshape(batch_size), device=device)

        # --- PPO updates (no predictor loss term) ---
        idx = np.arange(batch_size)
        pl_sum = vl_sum = ent_sum = 0.0
        n_mb = 0
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, batch_size, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]
                logits, v_ext_pred, v_int_pred = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                mb_adv = adv_t[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = (new_logp - old_logp_t[mb]).exp()
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                v_ext_loss = (v_ext_pred - ret_ext_t[mb]).pow(2).mean()
                v_int_loss = (v_int_pred - ret_int_t[mb]).pow(2).mean()
                value_loss = 0.5 * (v_ext_loss + v_int_loss)

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                pl_sum  += policy_loss.item()
                vl_sum  += value_loss.item()
                ent_sum += entropy.item()
                n_mb += 1

        global_step = update * batch_size
        if ep_returns:
            recent = float(np.mean(ep_returns[-20:]))
            print(f"update: {update:>4}  frames: {global_step:>8}  "
                  f"recent_mean_return: {recent:.1f}  episodes: {len(ep_returns)}  "
                  f"cells: {hasher.num_cells():>7}  int_r: {rew_int_buf.mean():.3f}  lr: {lr_now:.2e}")
        if args.wandb:
            log = {
                "global_step": global_step,
                "policy_loss":   pl_sum  / n_mb,
                "value_loss":    vl_sum  / n_mb,
                "entropy":       ent_sum / n_mb,
                "int_reward_mean":  float(rew_int_buf.mean()),
                "int_reward_std":   float(rew_int_buf.std()),
                "int_ret_rms_std":  float(np.sqrt(int_ret_rms.var)),
                "unique_cells":     hasher.num_cells(),
                "lr": lr_now,
            }
            if ep_returns:
                log["recent_mean_return"] = float(np.mean(ep_returns[-20:]))
            wandb.log(log, step=global_step)

    torch.save({
        "actor_critic":  model.state_dict(),
        "hash_matrix":   hasher.A,
        "counts":        dict(hasher.counts),
    }, SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}  ({hasher.num_cells():,} unique cells)")
