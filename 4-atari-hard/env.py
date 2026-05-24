"""Atari hard-exploration env setup.

Same preprocessing pipeline as 3-atari (frameskip 4, 84x84 grayscale,
framestack 4, sticky actions via the v5 env id), but without
LifeLossTerminalEnv: hard-exploration agents need uninterrupted long
trajectories so the intrinsic-reward chain can credit far-future novelty.
The underlying env still ends naturally on real game-over.

Default game is Montezuma's Revenge — the canonical hard-exploration
benchmark.
"""
import argparse
import sys

import ale_py
import gymnasium as gym
import numpy as np
import pygame
import torch

gym.register_envs(ale_py)


ENV_IDS = {
    "montezuma":   "ALE/MontezumaRevenge-v5",
    "pitfall":     "ALE/Pitfall-v5",
    "private_eye": "ALE/PrivateEye-v5",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_IDS), default="montezuma",
                   help="which hard-exploration Atari game to train on")
    p.add_argument("--render", action="store_true",
                   help="open a window during training (much slower)")
    p.add_argument("--test", action="store_true",
                   help="load the saved checkpoint and just play (no learning)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="override the auto-selected torch device")
    p.add_argument("--wandb", action="store_true",
                   help="log metrics to Weights & Biases")
    return p.parse_args()


def make_env(args):
    """Create a hard-exploration Atari env with the standard preprocessing.

    No FireResetEnv (Montezuma & friends don't need FIRE to launch) and no
    LifeLossTerminalEnv (we want full game-length episodes so intrinsic
    returns can chain across lives)."""
    env_id = ENV_IDS[args.env]
    env = gym.make(env_id, frameskip=1,
                   render_mode="human" if (args.render or args.test) else None)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def make_vec_env(args, n_envs):
    """Bundle n_envs copies of make_env into a SyncVectorEnv."""
    return gym.vector.SyncVectorEnv([lambda: make_env(args) for _ in range(n_envs)])


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
    """Replay episodes forever using the supplied action picker."""
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
