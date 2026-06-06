"""Atari hard-exploration env setup.

Two backends:

* `make_vec_env` returns an **envpool** vector env (C++, multithreaded). RND
  on Montezuma needs many parallel envs for the first key to be found by
  chance; SyncVectorEnv at 8-16 envs plateaus the policy in the first room
  forever. envpool gets us 64-128 envs at ~15k env-steps/s on an M3 with
  ~1.3 GB overhead, vs ~1.5k step/s and ballooning memory in Sync.

* `make_env` is still the gymnasium pipeline so `--test` can pop a window
  for human-mode rendering (envpool has no render mode).

Same preprocessing on both: frameskip 4, 84x84 grayscale, stack 4, sticky
actions (`repeat_action_probability=0.25`), full game-length episodes
(life loss does NOT terminate the episode — intrinsic returns need to
chain across deaths).

Default game is Montezuma's Revenge.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import types

# envpool eagerly imports a procgen submodule that links against homebrew
# Qt5 on macOS. We don't use procgen — stub both modules before import so
# the rest of envpool loads cleanly on arm64 Macs without brew install qt@5.
sys.modules["envpool.procgen.procgen_envpool"] = types.ModuleType("stub")
sys.modules["envpool.procgen.registration"] = types.ModuleType("stub")

import ale_py  # noqa: E402  (kept for the --test single-env path)
import envpool  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402
import torch  # noqa: E402

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Training utilities (reproducibility, structured logging, resumable runs).
# Shared by the training scripts in this folder. Plain training hygiene — kept
# here next to the other shared plumbing (make_env / parse_args / run_test_loop).
# ---------------------------------------------------------------------------

def seed_all(seed):
    """Seed python / numpy / torch for reproducibility (MPS is not bit-exact)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _atomic_save(state, path):
    """tmp -> rename so a crash mid-write never corrupts the checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


class RunLogger:
    """Optional run-directory outputs: metrics.jsonl, periodic / 5M-milestone /
    best checkpoints, resume, and a final.json summary. Inert when run_dir is
    None, so the training script still runs standalone."""

    def __init__(self, run_dir, ckpt_every):
        self.dir = run_dir
        self.ckpt_dir = os.path.join(run_dir, "ckpt") if run_dir else None
        self.ckpt_every = ckpt_every
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a", buffering=1) if run_dir else None
        self.t0, self.last_frames = time.time(), 0
        self.ckpt_last, self.ms_last, self.best = 0, 0, float("-inf")

    def log(self, frames, scalars):
        """Append one structured row (frames + sps + caller's scalars) to metrics.jsonl."""
        if not self.f:
            return
        now = time.time()
        sps = (frames - self.last_frames) / max(now - self.t0, 1e-9)
        self.f.write(json.dumps({"ts": round(now, 1), "frames": frames, "sps": round(sps, 1), **scalars}) + "\n")
        self.t0, self.last_frames = now, frames

    def resolve_resume(self, resume_arg):
        """'auto' -> run_dir/ckpt/latest.pt, else a path, else None."""
        if resume_arg == "auto" and self.ckpt_dir:
            cand = os.path.join(self.ckpt_dir, "latest.pt")
            return cand if os.path.exists(cand) else None
        if resume_arg and resume_arg != "auto":
            return resume_arg if os.path.exists(resume_arg) else None
        return None

    def checkpoint(self, frames, state_fn, gate=None):
        """Periodic 'latest', 5M-step milestone, and best-gate checkpoints.
        state_fn() builds the dict only when a save actually happens."""
        if not self.ckpt_dir or not self.ckpt_every:
            return
        if frames - self.ckpt_last >= self.ckpt_every:
            _atomic_save(state_fn(), os.path.join(self.ckpt_dir, "latest.pt"))
            self.ckpt_last = frames
        if frames - self.ms_last >= 5_000_000:
            _atomic_save(state_fn(), os.path.join(self.ckpt_dir, f"step_{frames // 1_000_000}M.pt"))
            self.ms_last = frames
        if gate is not None and gate > self.best:
            self.best = gate
            _atomic_save(state_fn(), os.path.join(self.ckpt_dir, "best.pt"))

    def finalize(self, frames, game_returns, state_fn, k=100):
        """Final 'latest' checkpoint + a final.json result summary."""
        if self.ckpt_dir:
            _atomic_save(state_fn(), os.path.join(self.ckpt_dir, "latest.pt"))
        if self.dir:
            tail = [float(x) for x in game_returns[-k:]]
            with open(os.path.join(self.dir, "final.json"), "w") as fh:
                json.dump({"frames_total": frames, "frames_unit": "agent_steps",
                           "gate_metric": "game_return_mean_lastK", "K": k,
                           "value_mean": statistics.fmean(tail) if tail else float("nan"),
                           "value_std": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
                           "episodes_counted": len(tail)}, fh, indent=1)
        if self.f:
            self.f.close()


# Gymnasium / ALE id (used by make_env / --test rendering) paired with the
# envpool task name (used by make_vec_env). envpool uses short names without
# the "ALE/" namespace.
ENV_IDS = {
    "montezuma":   ("ALE/MontezumaRevenge-v5", "MontezumaRevenge-v5"),
    "pitfall":     ("ALE/Pitfall-v5",          "Pitfall-v5"),
    "private_eye": ("ALE/PrivateEye-v5",       "PrivateEye-v5"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_IDS), default="montezuma",
                   help="which hard-exploration Atari game to train on")
    p.add_argument("--render", action="store_true",
                   help="open a window during training (single-env --test only)")
    p.add_argument("--test", action="store_true",
                   help="load the saved checkpoint and just play (no learning)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="override the auto-selected torch device")
    p.add_argument("--wandb", action="store_true",
                   help="log metrics to Weights & Biases")
    # --- reproducibility / run-management flags (all optional; omit them and the script runs as before) ---
    p.add_argument("--seed", type=int, default=None,
                   help="reproducibility seed (np/torch/envpool)")
    p.add_argument("--total-frames", type=int, default=None,
                   help="override the in-file TOTAL_FRAMES budget (agent steps)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="override the in-file N_ENVS (e.g. smaller for a smoke run)")
    p.add_argument("--run-dir", type=str, default=None,
                   help="run directory: write metrics.jsonl / ckpt / final.json here")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="periodic checkpoint interval in agent steps (resume-safe)")
    p.add_argument("--resume", type=str, default=None,
                   help="'auto' (run-dir/ckpt/latest.pt) or a checkpoint path")
    return p.parse_args()


def make_env(args):
    """Single gymnasium env with the standard Atari preprocessing.

    Used for `--test` rendering; envpool has no human render mode."""
    gym_id, _ = ENV_IDS[args.env]
    env = gym.make(gym_id, frameskip=1,
                   render_mode="human" if (args.render or args.test) else None)
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def make_vec_env(args, n_envs, seed=0):
    """envpool vector env. Returns (n_envs, 4, 84, 84) uint8 obs and accepts
    int32 actions of shape (n_envs,). `info` is a single dict of per-env
    arrays; `info["terminated"]` is the real game-over signal (lives==0).

    envpool's `observation_space` / `action_space` are already the single-env
    spaces (no `single_*` aliases like gymnasium vector envs)."""
    _, pool_id = ENV_IDS[args.env]
    return envpool.make_gymnasium(
        pool_id,
        num_envs=n_envs,
        seed=seed,
        stack_num=4,
        frame_skip=4,
        gray_scale=True,
        img_height=84, img_width=84,
        noop_max=30,
        episodic_life=False,          # life loss does not end the episode
        use_fire_reset=True,          # auto-FIRE on reset for games that need it
        repeat_action_probability=0.25,  # v5-equivalent sticky actions
        reward_clip=False,            # we sign-clip in the training loop
        max_episode_steps=27_000,     # standard Atari time limit
    )


def pick_device(arg="auto"):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def quit_if_window_closed(env):
    if not pygame.display.get_init():
        return
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            sys.exit()


def run_test_loop(env, get_action):
    """Replay episodes forever using the supplied action picker (single env)."""
    while True:
        obs, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            quit_if_window_closed(env)
            action = get_action(np.asarray(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        print(f"test score: {score}")
