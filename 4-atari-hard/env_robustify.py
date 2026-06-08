"""Robustification env plumbing — the backward algorithm of Go-Explore /
Salimans & Chen 2018 (arXiv:1812.03381), distilling a single demo into a
policy that works under sticky actions.

Faithful-but-small port of openai/atari-reset (+ the uber-research fork the
Nature paper used). Two pieces:

* `ReplayResetEnv` wraps one raw gymnasium ALE env. Each episode RESTORES to a
  point along the demo (`starting_point`) and the agent plays forward from
  there under sticky actions. The score counter is seeded with the demo's raw
  return up to that point, so "did the agent do as well as the demo from here"
  is a single comparison `score >= returns[-1]`.

* `ResetManager` owns the curriculum: starting points are staggered across the
  worker pool near the demo's end and marched BACKWARD as the agent succeeds
  (and nudged forward when it collapses). `max_starting_point -> 0` means the
  policy now plays the whole game from reset — the real progress metric.

Design notes (verified against atari-reset wrappers.py / ppo.py):
  1. Success = raw score (incl. demo prefix) >= demo's full return, minus an
     allowed deficit. Move rule (code, not paper): new max start = first index
     where cumsum(success_rate)/window >= move_threshold; else +nudge forward.
  2. lag kill: stay within `allowed_lag` steps of the demo's pace, compared to
     a windowed-min of returns (so faithful play through a negative reward
     isn't falsely killed).
  3. success kill: once as-good-as-demo, run exp(U(0,1)*7) extra steps then end
     with `random_reset=True` — the trainer masks GAE across this artificial
     boundary and the random length stops the agent timing the cutoff.
  4. warm-up replay of the last `reset_steps_ignored` demo actions through the
     step path warms the recurrent state; those transitions are `invalid` and
     masked from every loss.
  5. trained WITH sticky actions (Go-Explore, not S&C deterministic) so the
     policy is robust to the eval protocol; 0-30 no-ops when starting at reset.
"""
import pickle

import numpy as np


class StickyActionEnv:
    """repeat_action_probability applied BELOW frameskip — sticky at the raw
    action level, the standard v5 stochasticity. We build the ALE env with
    sticky 0 and add it here so the demo replay (which must be deterministic)
    can bypass it."""

    def __init__(self, p=0.25):
        self.p = p
        self.last = 0

    def reset(self):
        self.last = 0

    def filter(self, action, rng):
        if rng.random() < self.p:
            return self.last
        self.last = action
        return action


class ReplayResetEnv:
    """One raw ALE env that starts episodes from demo states. Not a gym env —
    the vectorized loop in 3-robustify.py drives it directly."""

    def __init__(self, demo, seed, *, sticky=0.25, allowed_lag=50,
                 allowed_score_deficit=0, reset_steps_ignored=0,
                 inc_entropy_threshold=100, noop_max=30, max_steps=400_000):
        import ale_py
        import gymnasium as gym
        gym.register_envs(ale_py)
        self.env = gym.make(demo["env_id"], frameskip=4,
                            repeat_action_probability=0.0,  # we add sticky ourselves
                            obs_type="grayscale").unwrapped
        self.ale = self.env.ale
        self.actions = demo["actions"]
        self.rewards = demo["rewards"]
        self.returns = demo["returns"]              # cumulative raw, return-to-here
        self.total_return = float(self.returns[-1])
        self.checkpoints = demo["checkpoints"]
        self.ckpt_nr = demo["checkpoint_action_nr"]
        self.n = len(self.actions)
        self.sticky = StickyActionEnv(sticky) if sticky > 0 else None
        self.allowed_lag = allowed_lag
        self.allowed_score_deficit = allowed_score_deficit
        self.reset_steps_ignored = reset_steps_ignored
        self.inc_entropy_threshold = inc_entropy_threshold
        self.noop_max = noop_max
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.starting_point = self.n - 1
        self.frac_sample = 0.2

    # --- frame preprocessing: 105x80 grayscale (atari-reset uses RGB; grayscale
    #     keeps us light and matches the rest of this repo). 4-stack handled in
    #     the trainer. Returns uint8 (105, 80). ---
    def _frame(self):
        import cv2
        g = self.ale.getScreenGrayscale()
        return cv2.resize(g, (80, 105), interpolation=cv2.INTER_AREA)

    def _restore_to(self, nr):
        """Restore the latest checkpoint at or before nr, replay demo actions
        up to nr (no sticky — deterministic), return the post-restore frame
        from a real act (never a stale post-restore read)."""
        ci = int(np.searchsorted(self.ckpt_nr, nr, side="right") - 1)
        ci = max(ci, 0)
        self.ale.restoreState(pickle.loads(self.checkpoints[ci]))
        replay_from = int(self.ckpt_nr[ci])
        last_frame = None
        for i in range(replay_from, nr):
            self.ale.act(int(self.actions[i]))
            last_frame = None  # frames during pure replay are not needed
        return last_frame

    def reset(self):
        # per-episode starting point: 0.8 at the pinned point, 0.2 uniform tail
        if self.rng.random() < self.frac_sample:
            nr = int(self.rng.integers(self.starting_point, self.n))
        else:
            nr = self.starting_point
        if self.sticky:
            self.sticky.reset()

        if nr <= 0:
            self.env.reset(seed=int(self.rng.integers(2 ** 31)))
            for _ in range(int(self.rng.integers(self.noop_max + 1))):
                self.ale.act(0)
            self.score = 0.0
            self.action_nr = 0
            self.start_nr = 0
        else:
            warm = max(nr - self.reset_steps_ignored, 0)
            self._restore_to(warm)
            self.score = float(self.returns[warm - 1]) if warm > 0 else 0.0
            self.action_nr = warm
            self.start_nr = nr  # success/entropy measured against the true start
        self.extra = 0
        # post-restore screen reads are STALE until the next act — take one real
        # NOOP to get a valid frame (not counted toward score/pace).
        frame, _ = self._step_raw(0, bookkeep=False)
        return frame

    def _step_raw(self, action, *, bookkeep=True):
        a = self.sticky.filter(action, self.rng) if self.sticky else action
        r = self.ale.act(int(a))
        if bookkeep:
            self.score += float(r)
            self.action_nr += 1
        return self._frame(), float(r)

    def step(self, action):
        frame, raw_r = self._step_raw(action)
        info = {"raw_reward": raw_r}
        done = False

        # success: as good as the demo from here
        if self.extra == 0 and self.score >= self.total_return - self.allowed_score_deficit:
            self.extra = int(np.exp(self.rng.random() * 7))  # 1..1096
        if self.extra > 0:
            self.extra -= 1
            if self.extra == 0:
                done = True
                info["random_reset"] = True
                info["as_good_as_demo"] = True

        # lag kill: fell behind the demo's pace (windowed-min, deficit-aware)
        t = self.action_nr
        if not done and t > self.allowed_lag and t < self.n:
            lo = max(t - self.allowed_lag, 0)
            hi = min(t + self.allowed_lag, self.n)
            threshold = float(self.returns[lo:hi].min()) - self.allowed_score_deficit
            if self.score < threshold:
                done = True

        if self.ale.game_over() or self.action_nr - self.start_nr >= self.max_steps:
            done = True
        info["increase_entropy"] = (self.action_nr < self.start_nr + self.inc_entropy_threshold)
        return frame, np.sign(raw_r), done, info  # clipped reward to the agent


class ResetManager:
    """Owns the shared curriculum across N envs. The trainer calls assign() once
    to stagger starting points, and update() each time a batch of episodes
    finishes to march max_starting_point backward."""

    def __init__(self, demo, n_envs, *, move_threshold=0.1, nudge=100, window=None):
        self.n = len(demo["actions"])
        self.n_envs = n_envs
        self.move_threshold = move_threshold
        self.nudge = nudge
        # window = the span of staggered starting points (atari-reset nrstartsteps).
        # The move target is move_threshold*window of cumulative success mass.
        self.window = window or max(n_envs, 32)
        self.max_starting_point = self.n - 1
        self.max_max = self.n - 1
        # latest success-rate per starting-point index
        self.success = np.zeros(self.n + 1, dtype=np.float64)

    def assign(self, envs):
        """Stagger envs across a window below max_starting_point."""
        per = max(self.window // max(self.n_envs, 1), 1)
        for i, e in enumerate(envs):
            e.starting_point = max(self.max_starting_point - i * per, 0)

    def record(self, starting_point, success):
        # exponential-ish freshening: latest wins (atari-reset keeps last rate)
        self.success[min(starting_point, self.n)] = float(success)

    def update(self, envs):
        """Move rule (atari-reset ResetManager.proc_infos): forward-cumsum the
        per-index success rates from index 0; the new max starting point is the
        FIRST index where the cumulative mass reaches move_threshold*window —
        i.e. march back as far as the practiced success band supports, no
        further. If the mass is never reached (success collapsed), nudge the
        curriculum forward (easier) by `nudge`."""
        tail = self.success[: self.max_starting_point + 1]
        csum = np.cumsum(tail)  # forward: mass accumulated up to each index
        hits = np.argwhere(csum >= self.move_threshold * self.window)
        if len(hits):
            new_max = int(hits[0][0])  # earliest index reaching the mass
            self.max_starting_point = max(min(new_max, self.max_starting_point), 0)
        else:
            self.max_starting_point = min(self.max_starting_point + self.nudge, self.max_max)
        self.assign(envs)
        return self.max_starting_point


def load_demo(path):
    with open(path, "rb") as f:
        return pickle.load(f)
