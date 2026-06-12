"""Extract a replayable demo from a Go-Explore Phase 1 run, for robustification.

The backward algorithm (3-robustify.py) needs, for the best trajectory the
archive found:
  - actions[]            the full action sequence (raw, frameskip-4 agent steps)
  - rewards[]            per-step RAW (unclipped) game reward
  - checkpoints[]        periodic pickled ALE states (restore points)
  - checkpoint_action_nr[]  the action index each checkpoint sits at

We get actions by walking the experience-log prev_id chain to the DONE cell
(reusing 2-go-explore's ExperienceLog), then replay them on the exact Phase-1
protocol (deterministic ALE, frameskip 4, sticky 0, seed 0) to capture rewards
and snapshots. The replay's cumulative score must equal the archived DONE
score — same determinism guarantee record-demo.py already relies on; a mismatch
aborts (a non-replayable demo is useless for robustification). Following the
papers we truncate just after the last reward and keep the shortest demo among
the best (here: the single DONE trajectory).

Usage:
  extract_demo.py --run-dir <ge-run-dir> --out <demo.pkl> [--ckpt-every 512]
"""
import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def _load_ge():
    """Import 2-go-explore.py (ExperienceLog/DONE_KEY) with env stubbed."""
    here = Path(__file__).resolve().parent
    import types
    stub = types.ModuleType("env_go_explore")
    stub.ENV_IDS = {"montezuma_goexplore": "ALE/MontezumaRevenge-v5"}
    for name in ("RunLogger", "make_restore_env", "parse_args"):
        setattr(stub, name, lambda *a, **k: None)
    sys.modules["env_go_explore"] = stub
    spec = importlib.util.spec_from_file_location("_ge", here / "2-go-explore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    del sys.modules["env_go_explore"]
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Go-Explore Phase 1 run dir")
    p.add_argument("--out", required=True, help="output demo.pkl path")
    p.add_argument("--ckpt-every", type=int, default=512,
                   help="snapshot an ALE restore point every N actions")
    p.add_argument("--max-rewards", type=int, default=0,
                   help="truncate just after the Kth nonzero reward instead of the last "
                        "(0 = last, default). Use 1 for a first-key-only easy demo — a much "
                        "shorter horizon for robustification to bootstrap on.")
    args = p.parse_args()

    ge = _load_ge()
    run_dir = Path(args.run_dir)

    import torch
    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "ckpt" / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    done = ckpt["archive"]["cells"].get(ge.DONE_KEY)
    if not done:
        sys.exit("[extract] no DONE cell — run has no end-of-episode trajectory")
    archived_score = done["score"]
    print(f"[extract] DONE score {archived_score:.0f}, traj_len {done['traj_len']:,}", flush=True)

    explog = ge.ExperienceLog(str(run_dir / "explog"))
    explog.load_state(ckpt["explog"])
    actions = explog.reconstruct_actions(done["traj_last"])
    assert len(actions) == done["traj_len"], (len(actions), done["traj_len"])

    # replay on the exact Phase-1 protocol, capturing rewards + periodic states
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make("ALE/MontezumaRevenge-v5", frameskip=4,
                   repeat_action_probability=0.0).unwrapped
    env.reset(seed=0)
    rewards, checkpoints, ckpt_nr = [], [], []
    score = 0.0
    last_reward_idx = -1
    for i, a in enumerate(actions):
        if i % args.ckpt_every == 0:
            checkpoints.append(pickle.dumps(env.ale.cloneState()))
            ckpt_nr.append(i)
        _, r, term, trunc, _ = env.step(int(a))
        rewards.append(float(r))
        score += float(r)
        if r != 0:
            last_reward_idx = i
        if term or trunc:
            break

    if score != archived_score:
        sys.exit(f"[extract] REPLAY MISMATCH {score} != {archived_score} — "
                 f"demo is not replayable, refusing to write")

    # truncate just after a reward (papers: start right before a reward; nothing
    # after it helps robustification). --max-rewards K cuts after the Kth reward
    # for a shorter, easier-to-bootstrap demo; default cuts after the last.
    reward_idxs = [i for i, r in enumerate(rewards) if r != 0.0]
    if args.max_rewards > 0 and len(reward_idxs) >= args.max_rewards:
        last_reward_idx = reward_idxs[args.max_rewards - 1]
    cut = last_reward_idx + 1
    actions, rewards = actions[:cut], rewards[:cut]
    checkpoints = [c for c, n in zip(checkpoints, ckpt_nr) if n < cut]
    ckpt_nr = [n for n in ckpt_nr if n < cut]

    demo = {
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "checkpoints": checkpoints,
        "checkpoint_action_nr": np.array(ckpt_nr, dtype=np.int64),
        "score": float(sum(rewards)),
        "returns": np.cumsum(rewards).astype(np.float32),  # return-to-here, raw
        "env_id": "ALE/MontezumaRevenge-v5",
        "protocol": {"frameskip": 4, "sticky": 0.0, "seed": 0},
        "source_run": str(run_dir),
        "ale_py": ale_py.__version__,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(demo, f)
    print(f"[extract] wrote {args.out}: {len(actions):,} actions, score {demo['score']:.0f}, "
          f"{len(checkpoints)} checkpoints, last reward @ {last_reward_idx}", flush=True)


if __name__ == "__main__":
    main()
