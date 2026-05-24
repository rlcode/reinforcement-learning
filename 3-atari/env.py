"""Shared Atari setup for DQN and PPO.

Picks Breakout or Pong via --env, applies the standard Atari preprocessing
(frameskip 4, 84x84 grayscale, framestack 4), and exposes the same
--render / --test CLI as the cartpole scripts.

Default device picks CUDA, falls back to MPS (Apple Silicon), then CPU.
"""
import argparse
import sys

import ale_py
import gymnasium as gym
import numpy as np
import pygame
import torch

gym.register_envs(ale_py)


# Breakout (and a few other games) require pressing FIRE to launch the ball
# after each reset / life loss. AtariPreprocessing only does NOOPs, so without
# this the agent wastes a lot of frames waiting for a random FIRE.
class FireResetEnv(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}


# Treats each life as its own episode for bootstrapping (so Q-targets / GAE don't
# value-chain across deaths) but only resets the real game when all lives are
# gone. Without this, every life loss triggers a full env.reset() — burning
# frames on noop_max + FIRE and breaking long-horizon credit assignment.
class LifeLossTerminalEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.game_over = terminated or truncated
        lives = info.get("lives", 0)
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        info["game_over"] = self.game_over
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.game_over:
            obs, info = self.env.reset(**kwargs)
        else:
            # Fake terminal from a life loss — advance one frame instead of
            # resetting so the game keeps its remaining lives.
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        return obs, info

ENV_IDS = {
    "breakout": "ALE/Breakout-v5",
    "pong":     "ALE/Pong-v5",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_IDS), default="breakout",
                   help="which Atari game to train on")
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
    """Create an Atari env with the standard preprocessing pipeline."""
    env_id = ENV_IDS[args.env]
    # frameskip=1 here because AtariPreprocessing applies its own.
    env = gym.make(env_id, frameskip=1,
                   render_mode="human" if (args.render or args.test) else None)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,  # handled by LifeLossTerminalEnv below
        grayscale_obs=True,
        scale_obs=False,        # keep uint8; we normalize in the model
    )
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = LifeLossTerminalEnv(env)
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
    """Exit cleanly when the user clicks the window's X.

    No-op on headless runs (no pygame display initialized).
    """
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
