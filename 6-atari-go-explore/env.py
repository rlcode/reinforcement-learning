"""Go-Explore env setup (restore-based exploration on raw gymnasium ALE).

Unlike the other Atari folders this one does NOT use envpool: Go-Explore's
exploration phase needs the emulator's save/restore API (ale.cloneState /
restoreState), which envpool does not expose. Each (worker) process owns a
single raw ALE env built by `make_restore_env`.

Protocol (Ecoffet et al. 2019/2021, exploration phase): fully deterministic —
frameskip 4, NO sticky actions, no no-ops, seed 0. Stochasticity only enters
in the (separate, later) robustification phase. The TimeLimit wrapper is
stripped (`.unwrapped`): its step counter is meaningless when episodes are
entered mid-trajectory via state restore.

★ Verified ALE pitfall (this machine, ale-py 0.11.2): right after
`restoreState`, `getRAM()` / screen reads still return the PRE-restore values
until the next `act()`. Callers must therefore derive cell keys only from
frames returned by `env.step()`, never from immediate post-restore reads.
"""
import argparse
import json
import os
import statistics
import time

import torch  # checkpoint serialization only — there is no neural net here


def _atomic_save(state, path):
    """tmp -> rename so a crash mid-write never corrupts the checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


class RunLogger:
    """Optional run-directory outputs: metrics.jsonl, periodic / milestone /
    best checkpoints, resume, and a final.json summary. Inert when run_dir is
    None, so the script still runs standalone.

    Same contract as 4-atari-hard/env.py with one change: milestone
    checkpoints fire every 50M frames instead of 5M — a Go-Explore checkpoint
    carries the whole archive (~0.5 GB at 50k cells), and a 500M-step run
    would otherwise pile up 100 of them."""

    MILESTONE_EVERY = 50_000_000

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
        """Periodic 'latest', 50M-step milestone, and best-gate checkpoints.
        state_fn() builds the dict only when a save actually happens."""
        if not self.ckpt_dir or not self.ckpt_every:
            return
        if frames - self.ckpt_last >= self.ckpt_every:
            _atomic_save(state_fn(), os.path.join(self.ckpt_dir, "latest.pt"))
            self.ckpt_last = frames
        if frames - self.ms_last >= self.MILESTONE_EVERY:
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


# Gymnasium / ALE ids. The "_goexplore" key marks a distinct benchmark
# protocol (deterministic, no sticky) — never cross-compare with the
# sticky-action `montezuma` numbers elsewhere in this repo.
ENV_IDS = {
    "montezuma_goexplore": "ALE/MontezumaRevenge-v5",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_IDS), default="montezuma_goexplore",
                   help="which game to explore")
    # --- reproducibility / run-management flags (harness run contract) ---
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the action RNG (the emulator itself is deterministic)")
    p.add_argument("--total-frames", type=int, default=None,
                   help="override the in-file TOTAL_FRAMES budget (agent steps actually executed)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="override the in-file N_WORKERS (parallel explorer processes)")
    p.add_argument("--run-dir", type=str, default=None,
                   help="run directory: write metrics.jsonl / ckpt / final.json here")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="periodic checkpoint interval in agent steps (resume-safe)")
    p.add_argument("--resume", type=str, default=None,
                   help="'auto' (run-dir/ckpt/latest.pt) or a checkpoint path")
    return p.parse_args()


def make_restore_env(env_key):
    """Single raw ALE env with clone/restore access.

    Imports live here (not module top) so harness-side tests can stub this
    module without pulling in ale_py. Returns the unwrapped env: TimeLimit's
    step counter would spuriously truncate restore-based exploration, and
    OrderEnforcing rejects step-after-restore patterns."""
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make(ENV_IDS[env_key], frameskip=4,
                   repeat_action_probability=0.0,  # deterministic — Phase 1 requirement
                   obs_type="grayscale").unwrapped
    env.reset(seed=0)  # canonical deterministic start; variation comes from action RNGs
    assert env.spec.kwargs.get("repeat_action_probability", None) == 0.0
    return env
