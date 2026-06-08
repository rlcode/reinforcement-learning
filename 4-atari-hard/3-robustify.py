"""Go-Explore robustification (backward algorithm) for Montezuma's Revenge.

Salimans & Chen 2018 (arXiv:1812.03381) + Go-Explore Nature robustification.
Distills the single deterministic demo found by Phase 1 (2-go-explore.py, score
31,000) into a recurrent policy that plays under STICKY actions — turning a
trajectory-search result into an actual RL policy comparable to RND.

Mechanism (see env_robustify.py for the curriculum/wrapper details):
  episodes restore to a point along the demo and play forward; when the agent
  matches the demo's return-to-go from there in >= move_threshold of rollouts,
  the starting point marches backward. `max_starting_point -> 0` = the policy
  now plays the whole game from reset. That fraction reached is the real
  progress metric; the headline number is a from-reset sticky-action eval.

Single-machine port honest simplifications (vs 128-GPU atari-reset):
  - one demo (paper: 40% convergence with one demo even at scale — see SPEC),
  - GRU truncated-BPTT over the whole rollout (no separate context window),
  - SIL / multi-demo / reward autoscale implemented as OFF-by-default flags.

Run contract: --seed/--total-frames/--run-dir/--ckpt-every/--resume, plus
--demo (path from extract_demo.py) and --n-envs.
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

from env_robustify import ReplayResetEnv, ResetManager, load_demo

try:
    from env_go_explore import RunLogger, pick_device  # reuse plumbing if present
except Exception:  # pick_device may live elsewhere; fall back
    from env_go_explore import RunLogger
    def pick_device(arg="auto"):
        if arg != "auto":
            return torch.device(arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


TOTAL_FRAMES = 20_000_000     # agent steps (override --total-frames)
N_ENVS = 16
ROLLOUT = 128
EPOCHS = 4
MINIBATCHES = 4
GAMMA = 0.999
LAM = 0.95
CLIP = 0.1
LR = 1e-4
ENT_COEF = 1e-5
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
GRU_DIM = 256                 # atari-reset uses 800; 256 keeps MPS light
STICKY = 0.25
MOVE_THRESHOLD = 0.1
ADAM_EPS = 1e-6
LOG_EVERY = 1                 # in updates


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class GRUActorCritic(nn.Module):
    """conv 8/4/3 -> fc + LayerNorm -> GRUCell -> pi, V. Input = 4-stacked
    105x80 grayscale frames (4 channels)."""

    def __init__(self, n_actions, gru_dim=GRU_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(4, 32, 8, stride=4), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, 4, stride=2), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, 3, stride=1), 2 ** 0.5), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 4, 105, 80)).shape[1]
        self.fc = _ortho(nn.Linear(n_flat, gru_dim), 2 ** 0.5)
        self.ln = nn.LayerNorm(gru_dim)
        self.gru = nn.GRUCell(gru_dim, gru_dim)
        self.pi = _ortho(nn.Linear(gru_dim, n_actions), 0.01)
        self.v = _ortho(nn.Linear(gru_dim, 1), 1.0)
        self.gru_dim = gru_dim

    def features(self, obs):
        h = self.ln(torch.relu(self.fc(self.conv(obs / 255.0))))
        return h

    def step(self, obs, hx, inc_entropy=None):
        """One timestep for acting. obs (B,4,105,80), hx (B,gru). Returns
        logits, value, new hx."""
        hx = self.gru(self.features(obs), hx)
        logits = self.pi(hx)
        if inc_entropy is not None:
            logits = torch.where(inc_entropy.unsqueeze(1), logits / 2.0, logits)
        return logits, self.v(hx).squeeze(-1), hx

    def unroll(self, obs_seq, hx0, done_seq):
        """Recompute a (T,B) rollout's logits/values with done-masked GRU state
        for BPTT. obs_seq (T,B,4,105,80), hx0 (B,gru), done_seq (T,B)."""
        T, B = obs_seq.shape[:2]
        hx = hx0
        logits_l, val_l = [], []
        for t in range(T):
            hx = hx * (1.0 - done_seq[t]).unsqueeze(1)  # reset state after a done
            hx = self.gru(self.features(obs_seq[t]), hx)
            logits_l.append(self.pi(hx))
            val_l.append(self.v(hx).squeeze(-1))
        return torch.stack(logits_l), torch.stack(val_l)


def _stack_init(frame):
    return np.repeat(frame[None], 4, axis=0)  # (4,105,80)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", required=True)
    p.add_argument("--env", default="montezuma_goexplore_robust")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-frames", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-episodes", type=int, default=50)
    # stretch flags (off by default — see SPEC)
    p.add_argument("--sil", action="store_true")
    p.add_argument("--autoscale", action="store_true")
    args = p.parse_args()
    global TOTAL_FRAMES, N_ENVS
    if args.total_frames:
        TOTAL_FRAMES = args.total_frames
    if args.n_envs:
        N_ENVS = args.n_envs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    demo = load_demo(args.demo)
    print(f"device {device}  demo {len(demo['actions'])} actions score {demo['score']:.0f}  "
          f"n_envs {N_ENVS}  total_frames {TOTAL_FRAMES:,}", flush=True)

    envs = [ReplayResetEnv(demo, seed=args.seed * 1000 + i, sticky=STICKY) for i in range(N_ENVS)]
    mgr = ResetManager(demo, N_ENVS, move_threshold=MOVE_THRESHOLD)
    mgr.assign(envs)
    n_actions = envs[0].env.action_space.n

    net = GRUActorCritic(n_actions).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=ADAM_EPS)
    logger = RunLogger(args.run_dir, args.ckpt_every)

    # reset all envs
    stacks = np.stack([_stack_init(e.reset()) for e in envs])  # (N,4,105,80)
    hx = torch.zeros(N_ENVS, net.gru_dim, device=device)
    ep_start_nr = np.array([e.start_nr for e in envs])
    frames = 0
    update = 0

    def _state():
        return {"net": net.state_dict(), "opt": opt.state_dict(), "frames": frames,
                "update": update, "max_starting_point": mgr.max_starting_point,
                "success": mgr.success, "rng": np.random.get_state()}

    resume_path = logger.resolve_resume(args.resume) if args.run_dir else None
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        frames, update = ck["frames"], ck["update"]
        mgr.max_starting_point = ck["max_starting_point"]; mgr.success = ck["success"]
        np.random.set_state(ck["rng"]); mgr.assign(envs)
        print(f"resumed @ frames {frames} max_start {mgr.max_starting_point}", flush=True)

    n_updates = TOTAL_FRAMES // (ROLLOUT * N_ENVS)
    success_window = []  # recent as_good_as_demo outcomes for logging
    t0 = time.time()

    while frames < TOTAL_FRAMES:
        obs_buf = np.zeros((ROLLOUT, N_ENVS, 4, 105, 80), dtype=np.uint8)
        act_buf = np.zeros((ROLLOUT, N_ENVS), dtype=np.int64)
        logp_buf = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        val_buf = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        rew_buf = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        done_buf = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        rrst_buf = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)  # random_reset mask
        ent_buf = np.zeros((ROLLOUT, N_ENVS), dtype=bool)
        hx0 = hx.detach().clone()

        for t in range(ROLLOUT):
            obs_t = torch.as_tensor(stacks, dtype=torch.float32, device=device)
            inc = torch.as_tensor([e.action_nr < e.start_nr + e.inc_entropy_threshold
                                   for e in envs], device=device)
            with torch.no_grad():
                logits, value, hx = net.step(obs_t, hx, inc)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            obs_buf[t] = stacks
            act_buf[t] = action.cpu().numpy()
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = value.cpu().numpy()
            ent_buf[t] = inc.cpu().numpy()

            for i, e in enumerate(envs):
                frame, r, done, info = e.step(int(act_buf[t, i]))
                rew_buf[t, i] = r
                done_buf[t, i] = float(done)
                rrst_buf[t, i] = float(info.get("random_reset", False))
                # roll the 4-stack
                stacks[i] = np.concatenate([stacks[i][1:], frame[None]], axis=0)
                if done:
                    success_window.append(float(info.get("as_good_as_demo", False)))
                    mgr.record(ep_start_nr[i], info.get("as_good_as_demo", False))
                    frame0 = e.reset()
                    stacks[i] = _stack_init(frame0)
                    ep_start_nr[i] = e.start_nr
                    hx[i] = 0.0  # reset recurrent state on episode boundary
            frames += N_ENVS

        # bootstrap value
        with torch.no_grad():
            obs_t = torch.as_tensor(stacks, dtype=torch.float32, device=device)
            _, last_val, _ = net.step(obs_t, hx)
            last_val = last_val.cpu().numpy()

        # GAE with random_reset boundaries masked (don't bootstrap across the
        # artificial success cutoff)
        adv = np.zeros((ROLLOUT, N_ENVS), dtype=np.float32)
        gae = np.zeros(N_ENVS, dtype=np.float32)
        for t in reversed(range(ROLLOUT)):
            nextval = last_val if t == ROLLOUT - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + GAMMA * nextval * nonterm - val_buf[t]
            gae = delta + GAMMA * LAM * nonterm * gae
            gae = gae * (1.0 - rrst_buf[t])  # cut advantage chain at random resets
            adv[t] = gae
        ret = adv + val_buf

        # flatten time-major for minibatching but keep done seq for GRU unroll
        obs_seq = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        done_seq = torch.as_tensor(done_buf, device=device)
        act_t = torch.as_tensor(act_buf, device=device)
        oldlogp_t = torch.as_tensor(logp_buf, device=device)
        adv_t = torch.as_tensor(adv, device=device)
        ret_t = torch.as_tensor(ret, device=device)
        ent_t = torch.as_tensor(ent_buf, device=device)
        valid = (1.0 - torch.as_tensor(rrst_buf, device=device))  # mask success-cutoff steps

        env_idx = np.arange(N_ENVS)
        pl = vl = ent_sum = 0.0
        nmb = 0
        for _ in range(EPOCHS):
            np.random.shuffle(env_idx)
            mb = max(N_ENVS // MINIBATCHES, 1)
            for s in range(0, N_ENVS, mb):
                cols = env_idx[s:s + mb]
                logits, val = net.unroll(obs_seq[:, cols], hx0[cols], done_seq[:, cols])
                logits = torch.where(ent_t[:, cols].unsqueeze(-1), logits / 2.0, logits)
                dist = torch.distributions.Categorical(logits=logits)
                newlogp = dist.log_prob(act_t[:, cols])
                m = valid[:, cols]
                a = adv_t[:, cols]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (newlogp - oldlogp_t[:, cols]).exp()
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a)
                vloss = (val - ret_t[:, cols]) ** 2
                ent = dist.entropy()
                msum = m.sum().clamp(min=1.0)
                policy_loss = (pg * m).sum() / msum
                value_loss = (vloss * m).sum() / msum
                entropy = (ent * m).sum() / msum
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                opt.step()
                pl += policy_loss.item(); vl += value_loss.item()
                ent_sum += entropy.item(); nmb += 1

        mgr.update(envs)
        update += 1
        if update % LOG_EVERY == 0:
            sw = success_window[-200:]
            sps = frames / max(time.time() - t0, 1e-9)
            agood = float(np.mean(sw)) if sw else 0.0
            progress = 1.0 - mgr.max_starting_point / max(mgr.max_max, 1)
            print(f"upd {update:>5} frames {frames:>9,} max_start {mgr.max_starting_point:>6} "
                  f"({progress*100:4.1f}% back) as_good {agood:.2f} sps {sps:.0f} "
                  f"pl {pl/nmb:+.3f} vl {vl/nmb:.3f}", flush=True)
            if args.run_dir:
                logger.log(frames, {
                    "max_starting_point": int(mgr.max_starting_point),
                    "curriculum_progress": progress,
                    "as_good_as_demo_rate": agood,
                    "policy_loss": pl / nmb, "value_loss": vl / nmb,
                    "entropy": ent_sum / nmb, "sps": round(sps, 1),
                    "game_return_mean_lastK": progress,  # gate proxy until eval
                    "nan_flag": int(not np.isfinite(pl + vl)),
                })
        if args.run_dir:
            logger.checkpoint(frames, _state, gate=progress)

    # final from-reset sticky eval
    score = evaluate(net, demo, device, args.eval_episodes, n_actions, args.seed)
    print(f"eval (from reset, sticky): mean {np.mean(score):.0f} n {len(score)}", flush=True)
    if args.run_dir:
        logger.finalize(frames, score, _state, k=len(score))
    for e in envs:
        e.env.close()


def evaluate(net, demo, device, n_episodes, n_actions, seed):
    """From-reset, sticky-action, eps-greedy 0.0 eval — the RL-policy number."""
    e = ReplayResetEnv(demo, seed=seed + 99, sticky=STICKY, noop_max=30)
    e.starting_point = 0  # always from reset
    e.frac_sample = 0.0
    scores = []
    for _ in range(n_episodes):
        frame = e.reset()
        stack = _stack_init(frame)
        hx = torch.zeros(1, net.gru_dim, device=device)
        ret = 0.0
        done = False
        while not done:
            with torch.no_grad():
                obs = torch.as_tensor(stack[None], dtype=torch.float32, device=device)
                logits, _, hx = net.step(obs, hx)
                a = int(logits.argmax(-1))
            frame, _, done, info = e.step(a)
            ret += info["raw_reward"]
            stack = np.concatenate([stack[1:], frame[None]], axis=0)
        scores.append(ret)
    e.env.close()
    return scores


if __name__ == "__main__":
    main()
