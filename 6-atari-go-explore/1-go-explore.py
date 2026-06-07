"""Go-Explore Phase 1 (exploration phase) for Montezuma's Revenge.

Ecoffet et al., 2019: "Go-Explore: a New Approach for Hard-Exploration
Problems" (arXiv:1901.10995); Nature 2021 version "First return, then
explore" (arXiv:2004.12919). No neural network: intrinsic-motivation methods
(RND etc.) suffer from detachment (forgetting promising frontiers) and
derailment (exploration noise breaking the return trip). Go-Explore fixes
both mechanically — remember everything in an archive, and RETURN exactly
via emulator state restore, then explore from there:

    archive: cell -> (best trajectory reaching it, emulator snapshot, score)
    loop:    sample cells (novelty-weighted) -> restore -> random exploration
             -> add/update cells reached

Design notes (verified against the official uber-research/go-explore code):

    1. Cell key = grayscale frame -> cv2.resize to 11x8 (INTER_AREA) ->
       quantize to 9 levels: floor(8 * p / 255). 88-byte key.
    2. Selection weight = 1 / sqrt(seen_times + 1) (Nature simplification);
       sampling WITH replacement, batch of 100; the virtual DONE cell is
       never selected.
    3. Exploration from a restored cell: up to K=100 agent steps, repeated
       random actions (keep current action w.p. 0.95 -> geometric runs,
       mean 20). Episode end = LIFE LOSS (or game over) -> the transition
       maps to the DONE cell and the exploration episode aborts.
    4. Archive accept rule: replace/insert iff score is higher, or equal
       score with a shorter trajectory. Scores are raw and unclipped.
       On update the cell's counters reset and its snapshot/trajectory are
       replaced; the *chosen* cell's chosen_since_new resets when anything
       new is found.
    5. Trajectories are not stored per cell: a global append-only experience
       log (prev_id linked list) + per-cell traj_last pointers reconstruct
       any cell's action sequence — this is the demo source for a future
       robustification phase, so the log is flushed to compressed chunks in
       the run dir rather than discarded.
    6. ★ ALE pitfall (machine-verified): post-restore RAM/screen reads are
       STALE until the next act. Cell keys come only from frames returned by
       env.step(); the lives baseline travels in cell metadata.
    7. frames axis = agent steps actually EXECUTED by workers (frameskip 4
       applied; the hypothetical "replay from start" steps are not counted),
       matching the harness budget/tier semantics.
    8. Phase-1 caveat: the score is a deterministic trajectory-search result,
       NOT an RL-policy score. Never compare against sticky-action RL
       numbers (e.g. the RND campaign) without this caveat.
"""
import multiprocessing as mp
import os
import pickle
import time

import cv2
import numpy as np

from env import ENV_IDS, RunLogger, make_restore_env, parse_args


TOTAL_FRAMES = 5_000_000       # agent steps executed (override with --total-frames)
BATCH_CELLS = 100              # cells sampled (with replacement) per iteration
EXPLORE_STEPS = 100            # K: max agent steps per exploration episode
ACTION_REPEAT_P = 0.95         # keep current action w.p. 0.95 (geometric, mean 20)
CELL_W, CELL_H = 11, 8         # downscale resolution (official fixed setting)
CELL_LEVELS = 8                # quantize to floor(8*p/255) -> values 0..8
N_WORKERS = 12                 # M4 Max 16 cores: leave headroom for master + OS
LOG_EVERY_BATCHES = 10         # metrics.jsonl cadence (~1M steps/100s at full speed)
EXPLOG_CHUNK = 1 << 22         # 4M entries per experience-log chunk (~40MB in RAM)
ROOM_RAM_BYTE = 3              # Montezuma current-room RAM index (diagnostic only)
DONE_KEY = (b"DONE", True)     # virtual end-of-episode cell (never sampled)


def cell_key(frame):
    """(210, 160) uint8 grayscale frame -> 88-byte archive key."""
    small = cv2.resize(frame, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    return ((small / 255.0) * CELL_LEVELS).astype(np.uint8).tobytes()


class Cell:
    """Archive entry. snapshot/lives describe the state AT this cell so a
    worker can restore and keep exploring; traj_last points into the
    experience log for trajectory reconstruction."""
    __slots__ = ("snapshot", "score", "traj_len", "traj_last",
                 "seen", "chosen", "chosen_since_new", "lives")

    def __init__(self, snapshot, score, traj_len, traj_last, lives):
        self.snapshot = snapshot
        self.score = score
        self.traj_len = traj_len
        self.traj_last = traj_last
        self.lives = lives
        self.seen = self.chosen = self.chosen_since_new = 0


class ExperienceLog:
    """Append-only step log as a prev_id linked list (design note 5).

    RAM holds only the active chunk; full chunks flush to
    <dir>/chunk_NNNNN.npz (compressed — rewards/dones are almost all zero).
    With dir=None (probes/tests) full chunks stay in RAM instead."""

    def __init__(self, log_dir, chunk_size=EXPLOG_CHUNK, ancestor_dir=None):
        self.dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.chunk_size = chunk_size
        self.ancestor = ancestor_dir   # explog dir of the run we resumed FROM:
        self.count = 0                 # chunks flushed before the resume live there
        self.n_flushed = 0
        self._ram_chunks = []          # dir=None mode only
        self._cache = {}               # chunk_idx -> loaded arrays (reconstruction)
        self._new_chunk()

    def _new_chunk(self):
        n = self.chunk_size
        self.prev = np.empty(n, dtype=np.int64)
        self.act = np.empty(n, dtype=np.uint8)
        self.rew = np.empty(n, dtype=np.float32)
        self.done = np.empty(n, dtype=np.uint8)
        self.fill = 0

    def append(self, prev_id, action, reward, done):
        i = self.fill
        self.prev[i], self.act[i], self.rew[i], self.done[i] = prev_id, action, reward, done
        self.fill += 1
        idx = self.count
        self.count += 1
        if self.fill == self.chunk_size:
            self._flush()
        return idx

    def _flush(self):
        arrays = {"prev": self.prev[:self.fill], "act": self.act[:self.fill],
                  "rew": self.rew[:self.fill], "done": self.done[:self.fill]}
        if self.dir:
            tmp = os.path.join(self.dir, f"chunk_{self.n_flushed:05d}.tmp")
            np.savez_compressed(tmp, **arrays)
            os.replace(f"{tmp}.npz", os.path.join(self.dir, f"chunk_{self.n_flushed:05d}.npz"))
        else:
            self._ram_chunks.append({k: v.copy() for k, v in arrays.items()})
        self.n_flushed += 1
        self._new_chunk()

    def _chunk_path(self, chunk_idx):
        """A flushed chunk lives in our own dir, or (after a cross-run-dir
        resume) in the ancestor run's explog dir."""
        own = os.path.join(self.dir, f"chunk_{chunk_idx:05d}.npz")
        if os.path.exists(own):
            return own
        if self.ancestor:
            anc = os.path.join(self.ancestor, f"chunk_{chunk_idx:05d}.npz")
            if os.path.exists(anc):
                return anc
        raise RuntimeError(f"explog chunk {chunk_idx} not found in {self.dir}"
                           + (f" or {self.ancestor}" if self.ancestor else ""))

    def _chunk(self, chunk_idx):
        if chunk_idx == self.n_flushed:
            return {"prev": self.prev, "act": self.act}
        if self.dir:
            if chunk_idx not in self._cache:
                z = np.load(self._chunk_path(chunk_idx))
                self._cache[chunk_idx] = {"prev": z["prev"], "act": z["act"]}
            return self._cache[chunk_idx]
        return self._ram_chunks[chunk_idx]

    def reconstruct_actions(self, last_id):
        """Walk the prev_id chain back to the root (-1); return actions in
        forward order. This is how demos are rebuilt for replay/Phase 2."""
        actions = []
        idx = last_id
        while idx >= 0:
            c = self._chunk(idx // self.chunk_size)
            off = idx % self.chunk_size
            actions.append(int(c["act"][off]))
            idx = int(c["prev"][off])
        return actions[::-1]

    def state(self):
        return {"count": self.count, "n_flushed": self.n_flushed,
                "chunk_size": self.chunk_size,
                "cur_prev": self.prev[:self.fill].copy(), "cur_act": self.act[:self.fill].copy(),
                "cur_rew": self.rew[:self.fill].copy(), "cur_done": self.done[:self.fill].copy()}

    def load_state(self, st):
        assert st["chunk_size"] == self.chunk_size, "explog chunk_size mismatch"
        if self.dir:  # flushed chunks must be reachable (own dir or ancestor's)
            self.n_flushed = st["n_flushed"]
            for i in range(st["n_flushed"]):
                self._chunk_path(i)  # raises loudly if a chunk is missing
        self.count, self.n_flushed = st["count"], st["n_flushed"]
        self._new_chunk()
        n = len(st["cur_prev"])
        self.prev[:n], self.act[:n] = st["cur_prev"], st["cur_act"]
        self.rew[:n], self.done[:n] = st["cur_rew"], st["cur_done"]
        self.fill = n


class Archive:
    """Cell store + novelty-weighted selection + the accept rule. All updates
    happen serially in the master process."""

    def __init__(self):
        self.cells = {}            # (key_bytes, done_bool) -> Cell
        self.rooms = set()         # diagnostic only (RAM byte 3)
        self.done_scores = []      # recent end-of-episode scores (logging)

    def seed_root(self, key, snapshot, lives):
        self.cells[(key, False)] = Cell(snapshot, 0.0, 0, -1, lives)

    @property
    def best_done_score(self):
        c = self.cells.get(DONE_KEY)
        return c.score if c else float("-inf")

    @property
    def max_archive_score(self):
        return max(c.score for k, c in self.cells.items() if k != DONE_KEY)

    def sample(self, n, rng):
        """n cells with replacement, p ∝ 1/sqrt(seen+1); DONE excluded."""
        keys = [k for k in self.cells if k != DONE_KEY]
        w = np.array([1.0 / np.sqrt(self.cells[k].seen + 1.0) for k in keys])
        csum = np.cumsum(w)
        picks = []
        for u in rng.random(n) * csum[-1]:
            k = keys[min(int(np.searchsorted(csum, u)), len(keys) - 1)]
            c = self.cells[k]
            c.chosen += 1
            c.chosen_since_new += 1
            picks.append((k, c))
        return picks

    def update_from_trajectory(self, chosen_key, res, explog):
        """Walk one exploration episode (master-side, serial): append to the
        experience log, accumulate raw score from the chosen cell's stored
        score, and apply the accept rule (design note 4)."""
        chosen = self.cells[chosen_key]
        cur_score = chosen.score
        cur_len = chosen.traj_len
        prev_id = chosen.traj_last
        found_new = False
        seen_this_episode = set()

        for i in range(res["n_steps"]):
            prev_id = explog.append(prev_id, res["actions"][i], res["rewards"][i], res["dones"][i])
            cur_score += float(res["rewards"][i])
            cur_len += 1
            done = bool(res["dones"][i])
            key = DONE_KEY if done else (res["keys"][i], False)

            cell = self.cells.get(key)
            if cell is None:
                self.cells[key] = Cell(res["snapshots"][i], cur_score, cur_len, prev_id,
                                       res["lives"][i])
                self.cells[key].seen = 1
                seen_this_episode.add(key)
                found_new = True
            else:
                if key not in seen_this_episode:
                    cell.seen += 1
                    seen_this_episode.add(key)
                if cur_score > cell.score or (cur_score == cell.score and cur_len < cell.traj_len):
                    cell.snapshot = res["snapshots"][i]
                    cell.score, cell.traj_len, cell.traj_last = cur_score, cur_len, prev_id
                    cell.lives = res["lives"][i]
                    cell.seen = cell.chosen = cell.chosen_since_new = 0  # reset_cell_on_update
                    found_new = True
            if done:
                self.done_scores.append(cur_score)
                break

        if found_new:
            chosen.chosen_since_new = 0
        self.rooms.update(res["rooms"])

    def state(self):
        return {"cells": {k: {"snapshot": c.snapshot, "score": c.score,
                              "traj_len": c.traj_len, "traj_last": c.traj_last,
                              "seen": c.seen, "chosen": c.chosen,
                              "chosen_since_new": c.chosen_since_new, "lives": c.lives}
                          for k, c in self.cells.items()},
                "rooms": sorted(self.rooms), "done_scores": self.done_scores[-200:]}

    def load_state(self, st):
        self.cells = {}
        for k, d in st["cells"].items():
            c = Cell(d["snapshot"], d["score"], d["traj_len"], d["traj_last"], d["lives"])
            c.seen, c.chosen, c.chosen_since_new = d["seen"], d["chosen"], d["chosen_since_new"]
            self.cells[k] = c
        self.rooms = set(st["rooms"])
        self.done_scores = list(st["done_scores"])


# ---------------------------------------------------------------------------
# Worker side. Top-level functions: mp 'spawn' re-imports this module, so the
# main body below stays behind the __main__ guard. Each worker owns one env.
# ---------------------------------------------------------------------------
_W = {}


def _worker_init(env_key):
    _W["env"] = make_restore_env(env_key)
    _W["ale"] = _W["env"].ale


def _explore_task(task):
    """task = (snapshot bytes | None for root reset, lives, k, seed).
    Restore -> up to k steps of repeated random actions; abort on life loss /
    game over. Returns per-step arrays (keys/snapshots for archive insert)."""
    snapshot, prev_lives, k, seed = task
    env, ale = _W["env"], _W["ale"]
    rng = np.random.default_rng(seed)
    if snapshot is None:
        env.reset(seed=0)
    else:
        ale.restoreState(pickle.loads(snapshot))
    # design note 6: NO reads here — the restored state is stale until we act

    n_actions = env.action_space.n
    actions, rewards, dones, keys, snapshots, lives_list, rooms = [], [], [], [], [], [], set()
    action = int(rng.integers(n_actions))
    for _ in range(k):
        if rng.random() > ACTION_REPEAT_P:
            action = int(rng.integers(n_actions))
        frame, reward, terminated, truncated, _ = env.step(action)
        lives = ale.lives()
        done = bool(terminated) or lives < prev_lives
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)
        keys.append(cell_key(frame))
        snapshots.append(pickle.dumps(ale.cloneState()))
        lives_list.append(lives)
        rooms.add(int(ale.getRAM()[ROOM_RAM_BYTE]))
        if done:
            break
        prev_lives = lives
    return {"n_steps": len(actions), "actions": actions, "rewards": rewards,
            "dones": dones, "keys": keys, "snapshots": snapshots,
            "lives": lives_list, "rooms": rooms}


if __name__ == "__main__":
    args = parse_args()
    if args.total_frames:
        TOTAL_FRAMES = args.total_frames
    if args.n_workers:
        N_WORKERS = args.n_workers
    seed = args.seed if args.seed is not None else 0
    rng = np.random.default_rng(seed)

    logger = RunLogger(args.run_dir, args.ckpt_every)
    explog_dir = os.path.join(args.run_dir, "explog") if args.run_dir else None
    # cross-run-dir resume: flushed explog chunks live next to the checkpoint
    # we resume from (the harness relaunches into a fresh run dir)
    resume_path = logger.resolve_resume(args.resume)
    ancestor = (os.path.join(os.path.dirname(os.path.dirname(resume_path)), "explog")
                if resume_path else None)
    explog = ExperienceLog(explog_dir, ancestor_dir=ancestor)
    archive = Archive()
    frames = 0
    batch = 0

    def _state_fn():
        return {"version": 1, "frames": frames, "batch": batch,
                "archive": archive.state(), "explog": explog.state(),
                "rng": rng.bit_generator.state}

    # --- resume or seed the root cell ---
    if resume_path:
        import torch
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        frames, batch = ckpt["frames"], ckpt["batch"]
        archive.load_state(ckpt["archive"])
        explog.load_state(ckpt["explog"])
        rng.bit_generator.state = ckpt["rng"]
        print(f"resumed from {resume_path}: frames={frames} batch={batch} "
              f"cells={len(archive.cells)} explog={explog.count}")
    else:
        # root cell from a fresh reset (reset obs is NOT stale — note 6 only
        # applies to restores)
        env0 = make_restore_env(args.env)
        frame0, _ = env0.reset(seed=0)
        archive.seed_root(cell_key(np.asarray(frame0)), pickle.dumps(env0.ale.cloneState()),
                          env0.ale.lives())
        env0.close()

    print(f"env: {args.env}  workers: {N_WORKERS}  total_frames: {TOTAL_FRAMES:,}  seed: {seed}")
    ctx = mp.get_context("spawn")
    t_start = time.time()
    with ctx.Pool(N_WORKERS, initializer=_worker_init, initargs=(args.env,)) as pool:
        while frames < TOTAL_FRAMES:
            picks = archive.sample(BATCH_CELLS, rng)
            tasks = [(c.snapshot, c.lives, EXPLORE_STEPS, int(rng.integers(2 ** 31)))
                     for _, c in picks]
            results = pool.map(_explore_task, tasks, chunksize=2)  # ordered -> deterministic
            for (key, _), res in zip(picks, results):
                archive.update_from_trajectory(key, res, explog)
                frames += res["n_steps"]
            batch += 1

            if batch % LOG_EVERY_BATCHES == 0:
                best = archive.best_done_score
                gate = best if best != float("-inf") else 0.0
                tail = archive.done_scores[-20:]
                print(f"batch {batch:>6}  frames {frames:>11,}  cells {len(archive.cells):>6}  "
                      f"best_done {gate:>8.0f}  max_arch {archive.max_archive_score:>8.0f}  "
                      f"rooms {len(archive.rooms):>3}")
                logger.log(frames, {
                    "game_return_mean_lastK": gate,  # semantics: best end-of-episode score (K=1)
                    "ep_return_mean": float(np.mean(tail)) if tail else 0.0,
                    "game_return_count": len(archive.done_scores),
                    "best_done_score": gate,
                    "max_archive_score": archive.max_archive_score,
                    "n_cells": len(archive.cells),
                    "rooms_found": len(archive.rooms),
                    "explog_entries": explog.count,
                    "batch": batch,
                    "nan_flag": 0,
                })
                logger.checkpoint(frames, _state_fn,
                                  gate=gate if gate > 0 else None)

    best = archive.best_done_score
    final_score = best if best != float("-inf") else 0.0
    hours = (time.time() - t_start) / 3600
    print(f"done: frames {frames:,}  cells {len(archive.cells)}  rooms {len(archive.rooms)}  "
          f"best_done {final_score:.0f}  ({hours:.2f}h)")
    # final.json value_mean = best end-of-episode score (the official Phase-1
    # metric; see targets.yaml montezuma_goexplore for the protocol caveat)
    logger.finalize(frames, [final_score], _state_fn, k=1)
