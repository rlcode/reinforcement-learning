"""Shared CartPole-v1 setup for the three cartpole algorithms.

Each algorithm file gets the same --render / --test CLI, the same env
construction, and the same test-mode loop — they differ only in how
they pick an action and how they load their checkpoint.
"""
import argparse
import sys

import gymnasium as gym
import numpy as np
import pygame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="show the cartpole window during training")
    parser.add_argument("--test", action="store_true",
                        help="load the saved checkpoint and just play (no learning)")
    return parser.parse_args()


def make_env(args):
    return gym.make("CartPole-v1",
                    render_mode="human" if (args.render or args.test) else None)


def quit_if_window_closed(env):
    """Exit cleanly when the user clicks the window's X.

    Gymnasium's classic_control renderer pumps pygame's internal event
    processing but doesn't act on QUIT, so without this nothing would
    happen on close.  Safe to call from headless runs too: when no
    display is initialized the function returns immediately.
    """
    if not pygame.display.get_init():
        return
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            sys.exit()


def run_test_loop(env, get_action):
    """Replay episodes forever using the supplied action picker.

    `get_action(state: np.ndarray) -> int`.
    """
    while True:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        score = 0
        while not done:
            quit_if_window_closed(env)
            action = get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.array(next_state, dtype=np.float32)
            score += reward
        print(f"test score: {score}")
