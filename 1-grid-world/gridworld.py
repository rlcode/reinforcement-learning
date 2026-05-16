"""Shared pygame grid-world environment.

Cross-platform (Mac/Windows/Linux) replacement for the original tkinter
environment used by the grid-world algorithms.

Layout:
  - 5x5 grid, 100 px per cell.
  - Agent (blue square)  starts at (0, 0).
  - Two obstacles (red triangles) at (1, 2) and (2, 1).  Reward = -100, terminal.
  - Goal     (green circle)       at (2, 2).             Reward = +100, terminal.
  - All other transitions give reward = 0.

Actions (matches the original code):
  0: up    1: down    2: left    3: right

The display is created lazily on the first call to render(), so the env
can also be used headlessly (e.g. for unit tests).
"""
import time

import numpy as np
import pygame

UNIT = 100        # pixel size of one grid cell
WIDTH = 5         # grid width  (cells)
HEIGHT = 5        # grid height (cells)
FPS_DELAY = 0.03  # sleep between frames during render()

# Colors (R, G, B).
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_LINE = (200, 200, 200)
AGENT_COLOR = (60, 120, 220)
OBSTACLE_COLOR = (220, 60, 60)
GOAL_COLOR = (60, 200, 100)
TEXT_COLOR = (40, 40, 40)


class Env:
    """Static-obstacle 5x5 grid-world used by tabular SARSA and Q-learning."""

    n_actions = 4
    action_space = ["u", "d", "l", "r"]

    def __init__(self, title="GridWorld"):
        self.title = title
        # Agent position in *grid* coordinates [x, y] (column, row).
        self.agent = [0, 0]
        # Static landmark positions, also in grid coords.
        self.obstacles = [[1, 2], [2, 1]]
        self.goal = [2, 2]

        # Pygame state initialized lazily so that render-less use stays fast.
        self._screen = None
        self._font = None
        self._clock = None

    # ---- core RL API -----------------------------------------------------

    def reset(self):
        self.agent = [0, 0]
        if self._screen is not None:
            self.render()
            time.sleep(0.3)
        return list(self.agent)

    def step(self, action):
        x, y = self.agent
        if action == 0 and y > 0:                # up
            y -= 1
        elif action == 1 and y < HEIGHT - 1:     # down
            y += 1
        elif action == 2 and x > 0:              # left
            x -= 1
        elif action == 3 and x < WIDTH - 1:      # right
            x += 1
        self.agent = [x, y]

        if self.agent == self.goal:
            return list(self.agent), 100, True
        if self.agent in self.obstacles:
            return list(self.agent), -100, True
        return list(self.agent), 0, False

    # ---- rendering -------------------------------------------------------

    def _ensure_display(self):
        if self._screen is not None:
            return
        pygame.init()
        pygame.display.set_caption(self.title)
        self._screen = pygame.display.set_mode((WIDTH * UNIT, HEIGHT * UNIT))
        self._font = pygame.font.SysFont(None, 18)
        self._clock = pygame.time.Clock()

    def render(self):
        self._ensure_display()
        # Drain event queue so the window stays responsive (and closeable).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self._screen.fill(WHITE)
        self._draw_grid()
        for ox, oy in self.obstacles:
            self._draw_triangle(ox, oy, OBSTACLE_COLOR)
        gx, gy = self.goal
        self._draw_circle(gx, gy, GOAL_COLOR)
        ax, ay = self.agent
        self._draw_square(ax, ay, AGENT_COLOR)
        # Overlay Q-values if the algorithm has pushed any.
        if getattr(self, "_q_overlay", None) is not None:
            self._draw_q_overlay(self._q_overlay)

        pygame.display.flip()
        time.sleep(FPS_DELAY)

    def _draw_grid(self):
        for c in range(WIDTH + 1):
            pygame.draw.line(self._screen, GRID_LINE,
                             (c * UNIT, 0), (c * UNIT, HEIGHT * UNIT))
        for r in range(HEIGHT + 1):
            pygame.draw.line(self._screen, GRID_LINE,
                             (0, r * UNIT), (WIDTH * UNIT, r * UNIT))

    def _cell_center(self, x, y):
        return (x * UNIT + UNIT // 2, y * UNIT + UNIT // 2)

    def _draw_square(self, x, y, color):
        size = int(UNIT * 0.65)
        cx, cy = self._cell_center(x, y)
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        pygame.draw.rect(self._screen, color, rect)

    def _draw_circle(self, x, y, color):
        cx, cy = self._cell_center(x, y)
        pygame.draw.circle(self._screen, color, (cx, cy), int(UNIT * 0.33))

    def _draw_triangle(self, x, y, color):
        cx, cy = self._cell_center(x, y)
        r = int(UNIT * 0.36)
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        pygame.draw.polygon(self._screen, color, points)

    # ---- value overlay (used by tabular agents) --------------------------

    def print_value_all(self, q_table):
        """Stash the Q-table; it gets drawn on the next render() call."""
        self._q_overlay = q_table

    def _draw_q_overlay(self, q_table):
        # q_table is a dict keyed by str([x, y]) with [up, down, left, right] values.
        for x in range(WIDTH):
            for y in range(HEIGHT):
                key = str([x, y])
                if key not in q_table:
                    continue
                qs = q_table[key]
                cx, cy = self._cell_center(x, y)
                # Position each action's value near its edge of the cell.
                offsets = [(0, -UNIT // 2 + 10),  # up    -> top center
                           (0,  UNIT // 2 - 10),  # down  -> bottom center
                           (-UNIT // 2 + 15, 0),  # left  -> left center
                           ( UNIT // 2 - 15, 0)]  # right -> right center
                for i, q in enumerate(qs):
                    text = self._font.render(f"{q:+.2f}", True, TEXT_COLOR)
                    rect = text.get_rect(center=(cx + offsets[i][0], cy + offsets[i][1]))
                    self._screen.blit(text, rect)
