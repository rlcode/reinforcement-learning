"""PPO + RND (Random Network Distillation) for hard-exploration Atari.

Burda et al., 2018: "Exploration by Random Network Distillation"
(arXiv:1810.12894). Vanilla PPO scores 0 on Montezuma's Revenge because the
first reward needs a ~100-step specific action sequence. RND adds an
intrinsic curiosity bonus that pulls the agent toward novel states:

    target_net (frozen random CNN)  : s -> f_target(s)
    predictor_net (learned)         : s -> f_pred(s)
    intrinsic_reward(s)             = || f_pred(s) - f_target(s) ||^2

Novel states have high prediction error (predictor never saw them); seen
states drop to near zero. Five things make RND actually work:

    1. RND input is the SINGLE last frame, not the 4-stack (avoids
       overfitting to stack correlations).
    2. That input is normalized with running mean/std and clipped to
       [-5, 5]; the stats are seeded by 50 rollouts of a random agent
       BEFORE training starts.
    3. The intrinsic stream uses two separate value heads and its own GAE
       that is NON-EPISODIC (next_nonterminal = 1 always) so curiosity can
       chain across deaths. The paper calls this the most impactful design
       choice.
    4. Intrinsic rewards are divided by the running std of their discounted
       returns -- scale only, no mean centering.
    5. The predictor is updated on only ~25% of each minibatch so it
       doesn't converge fast and kill the bonus.

Combined advantage: A = ext_coef * A_ext + int_coef * A_int.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import make_vec_env, parse_args, pick_device
import run_io


SAVE_PATH = "atari_ppo_rnd.pt"
TOTAL_FRAMES = 10_000_000
N_ENVS = 128                   # envpool: max trajectory diversity; swap-heavy on 8 GB
ROLLOUT_STEPS = 128            # batch = N_ENVS * ROLLOUT_STEPS = 16384
EPOCHS = 4
MINIBATCH_SIZE = 2048          # 8 minibatches per epoch at batch=16384
CLIP_COEF = 0.1
GAMMA_EXT = 0.999              # sparse-reward games need long horizons
GAMMA_INT = 0.99               # curiosity is short-horizon by nature
GAE_LAMBDA = 0.95
LR = 1e-4
EXT_COEF = 2.0
INT_COEF = 1.0
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01            # paper uses 0.001 (assumes 1B+ frames); at 10M frames the
                               # policy entropy collapsed by ~50% before any breakthrough,
                               # so we use the standard PPO value to keep exploration alive
MAX_GRAD_NORM = 0.5
PREDICTOR_UPDATE_PROPORTION = 0.05  # paper default 0.25 saturates the predictor in ~500k
                                     # frames at our scale, killing the intrinsic signal; slow it
                                     # down 5x so novelty stays alive long enough to matter
OBS_NORM_WARMUP_ROLLOUTS = 50  # 50 * ROLLOUT_STEPS random transitions before training


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


# Same Nature CNN trunk as 3-atari/2-ppo.py, but with two value heads.
class ActorCriticRND(nn.Module):
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
        self.value_int = _ortho(nn.Linear(512, 1), 1.0)  # second head: intrinsic returns

    def forward(self, x):
        h = self.conv(x.float() / 255.0)
        return self.policy(h), self.value_ext(h).squeeze(-1), self.value_int(h).squeeze(-1)


# RND target/predictor share the same conv backbone with LeakyReLU
# (paper section 4.1). Input is a SINGLE 84x84 frame normalized to ~[-5, 5].
def _rnd_conv():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(),
        nn.Flatten(),
    )


class RNDTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _rnd_conv()
        self.fc = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x):
        return self.fc(self.conv(x))


class RNDPredictor(nn.Module):
    """Slightly deeper than target — two extra ReLU FCs so it has the
    capacity to actually fit the target's random projection."""
    def __init__(self):
        super().__init__()
        self.conv = _rnd_conv()
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        return self.head(self.conv(x))


class RunningMeanStd:
    """Welford / Chan parallel algorithm. Used for both obs (last frame)
    and intrinsic-return scaling."""
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


def normalize_obs_for_rnd(frame, obs_rms):
    """frame: (..., 84, 84) uint8 or float. Return float32 (..., 1, 84, 84)
    centered/scaled by obs_rms and clipped to [-5, 5] per paper."""
    x = frame.astype(np.float32)
    x = (x - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    x = np.clip(x, -5.0, 5.0)
    return x.astype(np.float32)


def compute_gae(rewards, values, nonterminals, last_value, gamma, lam):
    """Generic GAE. Pass nonterminals=1-dones for the extrinsic (episodic)
    stream, or all-ones for the intrinsic (non-episodic) stream."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminals[t] - values[t]
        gae = delta + gamma * lam * nonterminals[t] * gae
        advantages[t] = gae
    return advantages, advantages + values


def warmup_obs_rms(envs, obs_rms, n_steps):
    """Step a random agent so obs running stats are realistic before training.
    Without this, the first intrinsic rewards are wildly scaled and the
    predictor never recovers.

    Updates obs_rms incrementally each step to avoid building a multi-GB list
    of frames when N_ENVS is large."""
    print(f"warmup: stepping random agent for {n_steps} env steps to seed obs RMS...")
    obs, _ = envs.reset()
    n_actions = envs.action_space.n
    for _ in range(n_steps):
        actions = np.random.randint(0, n_actions, size=envs.num_envs, dtype=np.int32)
        next_obs, _, _, _, _ = envs.step(actions)
        # FrameStackObservation gives (n_envs, 4, 84, 84); update RMS with newest frames only.
        obs_rms.update(np.asarray(next_obs)[:, -1, :, :])
        obs = next_obs
    print(f"  obs_rms seeded with {obs_rms.count:.0f} samples, "
          f"mean={obs_rms.mean.mean():.2f}, std={np.sqrt(obs_rms.var).mean():.2f}")
    return obs


if __name__ == "__main__":
    args = parse_args()
    # 하네스 구동 오버라이드 (없으면 in-file 상수 유지)
    if args.n_envs:
        N_ENVS = args.n_envs
    if args.total_frames:
        TOTAL_FRAMES = args.total_frames
    if args.seed is not None:
        run_io.seed_all(args.seed)
    device = pick_device(args.device)
    envs = make_vec_env(args, N_ENVS, seed=args.seed or 0)
    n_actions = envs.action_space.n
    obs_shape = envs.observation_space.shape  # (4, 84, 84) — envpool single-env spec

    model = ActorCriticRND(n_actions).to(device)
    rnd_target = RNDTarget().to(device)
    rnd_predictor = RNDPredictor().to(device)
    for p in rnd_target.parameters():
        p.requires_grad_(False)
    rnd_target.eval()

    optimizer = optim.Adam(
        list(model.parameters()) + list(rnd_predictor.parameters()),
        lr=LR, eps=1e-5,
    )

    writer = run_io.RunWriter(args.run_dir, ckpt_every=args.ckpt_every)
    start_update = 0  # resume이 갱신

    if args.wandb:
        import wandb
        wandb.init(project="rl-atari-hard-ppo-rnd", config={
            "env": args.env, "n_envs": N_ENVS, "rollout_steps": ROLLOUT_STEPS,
            "total_frames": TOTAL_FRAMES, "epochs": EPOCHS,
            "minibatch_size": MINIBATCH_SIZE, "clip_coef": CLIP_COEF,
            "gamma_ext": GAMMA_EXT, "gamma_int": GAMMA_INT, "gae_lambda": GAE_LAMBDA,
            "lr": LR, "ext_coef": EXT_COEF, "int_coef": INT_COEF,
            "value_coef": VALUE_COEF, "entropy_coef": ENTROPY_COEF,
            "predictor_update_proportion": PREDICTOR_UPDATE_PROPORTION,
            "obs_norm_warmup_rollouts": OBS_NORM_WARMUP_ROLLOUTS,
        })

    print(f"device: {device},  env: {args.env},  actions: {n_actions},  n_envs: {N_ENVS}")

    obs_rms = RunningMeanStd(shape=(84, 84))
    int_ret_rms = RunningMeanStd(shape=())

    obs = warmup_obs_rms(envs, obs_rms, n_steps=OBS_NORM_WARMUP_ROLLOUTS * ROLLOUT_STEPS)

    batch_size = ROLLOUT_STEPS * N_ENVS
    n_updates = TOTAL_FRAMES // batch_size
    ep_returns_per_env = np.zeros(N_ENVS, dtype=np.float32)
    ep_returns = []  # extrinsic (raw, unclipped) per-game returns
    int_filter = np.zeros(N_ENVS, dtype=np.float64)  # discounted intrinsic returns for RMS

    def _state_fn():
        """체크포인트 상태 — 정규화기·int_filter·update까지 포함해야 resume이 거짓말 안 한다."""
        return {
            "actor_critic": model.state_dict(),
            "rnd_predictor": rnd_predictor.state_dict(),
            "rnd_target": rnd_target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms": {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
            "int_ret_rms": {"mean": int_ret_rms.mean, "var": int_ret_rms.var, "count": int_ret_rms.count},
            "int_filter": int_filter, "update": update,
            "ep_returns": ep_returns[-200:],
        }

    # --- resume (정규화기·optimizer·update 카운터 복원) ---
    resume_path = run_io.resolve_resume(args.resume, args.run_dir)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["actor_critic"])
        rnd_predictor.load_state_dict(ckpt["rnd_predictor"])
        rnd_target.load_state_dict(ckpt["rnd_target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for rms, saved in [(obs_rms, ckpt["obs_rms"]), (int_ret_rms, ckpt["int_ret_rms"])]:
            rms.mean, rms.var, rms.count = saved["mean"], saved["var"], saved["count"]
        int_filter = ckpt["int_filter"]
        ep_returns = list(ckpt["ep_returns"])
        start_update = ckpt["update"]
        print(f"resumed from {resume_path} at update {start_update}")

    global_step = start_update * batch_size  # 루프가 안 돌아도 final()이 참조 가능
    for update in range(start_update + 1, n_updates + 1):
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

            # Intrinsic reward: predict on the next frame's last channel.
            next_last = np.asarray(next_obs)[:, -1, :, :]
            obs_rms.update(next_last)
            with torch.no_grad():
                x = normalize_obs_for_rnd(next_last, obs_rms)  # (n_envs, 84, 84)
                x_t = torch.as_tensor(x, device=device).unsqueeze(1)  # (n_envs, 1, 84, 84)
                err = (rnd_predictor(x_t) - rnd_target(x_t)).pow(2).mean(dim=-1)
                rew_int_buf[t] = err.cpu().numpy()

            for i in range(N_ENVS):
                if done[i]:
                    ep_returns.append(float(ep_returns_per_env[i]))
                    ep_returns_per_env[i] = 0.0
            obs = next_obs

        # --- Intrinsic reward normalization (running std of discounted intrinsic returns) ---
        # Walk forward through the rollout, accumulating per-env discounted returns,
        # update int_ret_rms with all visited values, then scale rew_int by current std.
        for t in range(ROLLOUT_STEPS):
            int_filter = int_filter * GAMMA_INT + rew_int_buf[t]
            int_ret_rms.update(int_filter.copy())
        rew_int_buf = rew_int_buf / np.sqrt(int_ret_rms.var + 1e-8)

        # --- Dual GAE: extrinsic episodic, intrinsic non-episodic ---
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

        # Flatten (T, N_ENVS, ...) -> (T*N_ENVS, ...)
        obs_t        = torch.as_tensor(obs_buf.reshape(batch_size, *obs_shape), device=device)
        act_t        = torch.as_tensor(act_buf.reshape(batch_size), device=device)
        old_logp_t   = torch.as_tensor(logp_buf.reshape(batch_size), device=device)
        adv_t        = torch.as_tensor(adv_combined.reshape(batch_size), device=device)
        ret_ext_t    = torch.as_tensor(ret_ext.reshape(batch_size), device=device)
        ret_int_t    = torch.as_tensor(ret_int.reshape(batch_size), device=device)
        # RND predictor input: precompute normalized last frames once per rollout.
        last_frames = obs_buf[:, :, -1, :, :].reshape(batch_size, 84, 84)
        rnd_in_t    = torch.as_tensor(
            normalize_obs_for_rnd(last_frames, obs_rms), device=device,
        ).unsqueeze(1)  # (batch, 1, 84, 84)

        # --- PPO + RND updates ---
        idx = np.arange(batch_size)
        pl_sum = vl_sum = ent_sum = rnd_sum = kl_sum = 0.0
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

                logratio = new_logp - old_logp_t[mb]
                ratio = logratio.exp()
                with torch.no_grad():  # 진단용 approx KL (트립와이어 — watch.py)
                    kl_sum += ((ratio - 1) - logratio).mean().item()
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                v_ext_loss = (v_ext_pred - ret_ext_t[mb]).pow(2).mean()
                v_int_loss = (v_int_pred - ret_int_t[mb]).pow(2).mean()
                value_loss = 0.5 * (v_ext_loss + v_int_loss)

                # Predictor MSE on a random ~25% slice of the minibatch.
                pred = rnd_predictor(rnd_in_t[mb])
                with torch.no_grad():
                    tgt = rnd_target(rnd_in_t[mb])
                per_sample = (pred - tgt).pow(2).mean(dim=-1)
                keep = (torch.rand_like(per_sample) < PREDICTOR_UPDATE_PROPORTION).float()
                rnd_loss = (per_sample * keep).sum() / keep.sum().clamp(min=1.0)

                loss = (policy_loss
                        + VALUE_COEF * value_loss
                        - ENTROPY_COEF * entropy
                        + rnd_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rnd_predictor.parameters()),
                    MAX_GRAD_NORM,
                )
                optimizer.step()

                pl_sum  += policy_loss.item()
                vl_sum  += value_loss.item()
                ent_sum += entropy.item()
                rnd_sum += rnd_loss.item()
                n_mb += 1

        global_step = update * batch_size
        if ep_returns:
            recent = float(np.mean(ep_returns[-20:]))
            print(f"update: {update:>4}  frames: {global_step:>8}  "
                  f"recent_mean_return: {recent:.1f}  episodes: {len(ep_returns)}  "
                  f"int_reward_mean: {rew_int_buf.mean():.3f}  lr: {lr_now:.2e}")

        # --- 하네스 계약: metrics.jsonl (watch.py 트립와이어 소비) + 주기 체크포인트 ---
        loss_mean = (pl_sum + vl_sum + rnd_sum) / max(n_mb, 1)
        writer.log(global_step, {
            "game_return_mean_lastK": float(np.mean(ep_returns[-100:])) if ep_returns else 0.0,
            "ep_return_mean": float(np.mean(ep_returns[-20:])) if ep_returns else 0.0,
            "game_return_count": len(ep_returns),
            "entropy": ent_sum / max(n_mb, 1),
            "approx_kl": kl_sum / max(n_mb, 1),
            "policy_loss": pl_sum / max(n_mb, 1),
            "value_loss": vl_sum / max(n_mb, 1),
            "predictor_loss": rnd_sum / max(n_mb, 1),
            "int_rew_mean": float(rew_int_buf.mean()),
            "int_rew_std": float(rew_int_buf.std()),
            "lr": lr_now,
            "nan_flag": int(not np.isfinite(loss_mean)),
        })
        writer.maybe_ckpt(global_step, _state_fn,
                          gate_metric=float(np.mean(ep_returns[-100:])) if ep_returns else None)
        if args.wandb:
            log = {
                "global_step": global_step,
                "policy_loss":   pl_sum  / n_mb,
                "value_loss":    vl_sum  / n_mb,
                "entropy":       ent_sum / n_mb,
                "rnd_loss":      rnd_sum / n_mb,
                "int_reward_mean":  float(rew_int_buf.mean()),
                "int_reward_std":   float(rew_int_buf.std()),
                "obs_rms_mean":     float(obs_rms.mean.mean()),
                "obs_rms_std":      float(np.sqrt(obs_rms.var).mean()),
                "int_ret_rms_std":  float(np.sqrt(int_ret_rms.var)),
                "lr": lr_now,
            }
            if ep_returns:
                log["recent_mean_return"] = float(np.mean(ep_returns[-20:]))
            wandb.log(log, step=global_step)

    # 종료: 표준 체크포인트(latest) + 게이트 수치(final.json) + 기존 SAVE_PATH 호환
    writer.save_final_ckpt(_state_fn)
    writer.final(global_step, ep_returns, k=100)
    torch.save({
        "actor_critic": model.state_dict(),
        "rnd_predictor": rnd_predictor.state_dict(),
        "rnd_target": rnd_target.state_dict(),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var":  obs_rms.var,
    }, SAVE_PATH)
    print(f"Saved trained model to {SAVE_PATH}")
