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

UNIT = 100        # pixel size of one grid cell (tabular Env)
DYN_UNIT = 50     # pixel size of one grid cell (DynamicEnv — smaller, more cells fit)
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


class DynamicEnv:
    """5x5 grid-world with moving obstacles, used by Deep SARSA and REINFORCE.

    Differences from the static Env above:
      - 3 obstacles instead of 2; they jump left/right one cell every 2nd step.
      - Goal at (4, 4); obstacles initialize at (0,1), (1,2), (2,3).
      - Episode terminates only on goal — hitting an obstacle costs -1 but
        the episode continues (matches the original tkinter env).
      - Reward includes an optional per-step penalty (REINFORCE uses -0.1
        to encourage shorter paths; Deep SARSA uses 0).
      - State is a 15-dim relative encoding (see _get_state).

    Action mapping (matches the original deep-grid-world code):
      0: up    1: down    2: right    3: left
    """

    n_actions = 4
    action_space = ["u", "d", "r", "l"]
    state_size = 15

    HUD_HEIGHT = 32  # pixels reserved at the top of the window for the HUD

    def __init__(self, title="DynamicGridWorld", step_penalty=0.0, render_mode="human"):
        self.title = title
        self.step_penalty = step_penalty
        # render_mode=None disables display (useful for tests / headless training).
        self.render_mode = render_mode
        # Agent in grid coords.
        self.agent = [0, 0]
        # Obstacles: each is dict(state=[x,y], direction=+1/-1, reward=-1).
        # direction = -1 means "next move = right" (matches original code).
        self.obstacles_init = [[0, 1], [1, 2], [2, 3]]
        self.goal = [4, 4]
        self.obstacles = []
        self.counter = 0

        # HUD / feedback state.
        self.episode = 0
        self.score = 0.0
        # Number of remaining frames to flash the agent red after a -1 hit.
        self._hit_flash = 0

        self._screen = None
        self._hud_font = None
        self._popup_font = None
        self._clock = None

    # ---- core RL API -----------------------------------------------------

    def reset(self):
        # Episode count goes up at every reset *except* the very first.
        if self.counter > 0 or self.score != 0.0:
            self.episode += 1
        self.agent = [0, 0]
        self.counter = 0
        self.score = 0.0
        self._hit_flash = 0
        self.obstacles = [{"state": list(p), "direction": -1} for p in self.obstacles_init]
        if self._screen is not None:
            self.render()
            time.sleep(0.3)
        return self._get_state()

    def step(self, action):
        self.counter += 1
        # Render every step (matches the original tkinter env's behavior of
        # calling self.render() at the top of step()).  Skips display setup
        # if you never opened a display in this process.
        self.render()
        # Obstacles move every other step (matches the original env).
        if self.counter % 2 == 1:
            self._move_obstacles()

        x, y = self.agent
        if action == 0 and y > 0:                # up
            y -= 1
        elif action == 1 and y < HEIGHT - 1:     # down
            y += 1
        elif action == 2 and x < WIDTH - 1:      # right
            x += 1
        elif action == 3 and x > 0:              # left
            x -= 1
        self.agent = [x, y]

        reward, done = self._reward_and_done()
        reward -= self.step_penalty
        self.score += reward
        # Flash agent red for a few frames when it lands on an obstacle.
        if reward < -self.step_penalty:
            self._hit_flash = 4
        return self._get_state(), reward, done

    def _reward_and_done(self):
        if self.agent == self.goal:
            return 1.0, True
        # Reward stacks if agent lands on multiple obstacles in same cell
        # (the original env did this too — usually only one obstacle per cell).
        r = 0.0
        for obs in self.obstacles:
            if obs["state"] == self.agent:
                r -= 1.0
        return r, False

    def _move_obstacles(self):
        for obs in self.obstacles:
            x, y = obs["state"]
            # Bounce at the grid edges.
            if x == WIDTH - 1:
                obs["direction"] = 1
            elif x == 0:
                obs["direction"] = -1
            # direction == -1  => move right; direction == 1  => move left.
            x += 1 if obs["direction"] == -1 else -1
            obs["state"] = [x, y]

    def _get_state(self):
        """15-dim relative encoding (matches the original env).

        For each obstacle (3x):  [dx, dy, -1, direction]   (4 dims)
        For the goal      (1x):  [dx, dy,  1]              (3 dims)
        Total: 3 * 4 + 3 = 15.
        """
        ax, ay = self.agent
        s = []
        for obs in self.obstacles:
            ox, oy = obs["state"]
            s.append(ox - ax)
            s.append(oy - ay)
            s.append(-1)
            s.append(obs["direction"])
        gx, gy = self.goal
        s.append(gx - ax)
        s.append(gy - ay)
        s.append(1)
        return s

    # ---- rendering -------------------------------------------------------

    def _ensure_display(self):
        if self._screen is not None:
            return
        pygame.init()
        pygame.display.set_caption(self.title)
        self._screen = pygame.display.set_mode(
            (WIDTH * DYN_UNIT, HEIGHT * DYN_UNIT + self.HUD_HEIGHT)
        )
        self._hud_font = pygame.font.SysFont(None, 22)
        self._popup_font = pygame.font.SysFont(None, 28)
        self._clock = pygame.time.Clock()

    def render(self):
        if self.render_mode is None:
            return
        self._ensure_display()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self._screen.fill(WHITE)
        hud = self.HUD_HEIGHT  # vertical offset for the grid

        # --- HUD bar at the top: "Episode: N    Score: X.X" ----------------
        pygame.draw.rect(self._screen, (30, 30, 30),
                         pygame.Rect(0, 0, WIDTH * DYN_UNIT, hud))
        hud_text = f"Episode: {self.episode}    Score: {self.score:+.1f}"
        surf = self._hud_font.render(hud_text, True, (240, 240, 240))
        self._screen.blit(surf, (8, (hud - surf.get_height()) // 2))

        # --- Grid lines (offset down by HUD) -------------------------------
        for c in range(WIDTH + 1):
            pygame.draw.line(self._screen, GRID_LINE,
                             (c * DYN_UNIT, hud),
                             (c * DYN_UNIT, hud + HEIGHT * DYN_UNIT))
        for r in range(HEIGHT + 1):
            pygame.draw.line(self._screen, GRID_LINE,
                             (0, hud + r * DYN_UNIT),
                             (WIDTH * DYN_UNIT, hud + r * DYN_UNIT))

        # --- Goal ----------------------------------------------------------
        gx, gy = self.goal
        cx, cy = gx * DYN_UNIT + DYN_UNIT // 2, hud + gy * DYN_UNIT + DYN_UNIT // 2
        pygame.draw.circle(self._screen, GOAL_COLOR, (cx, cy), int(DYN_UNIT * 0.33))

        # --- Obstacles -----------------------------------------------------
        for obs in self.obstacles:
            ox, oy = obs["state"]
            cx, cy = ox * DYN_UNIT + DYN_UNIT // 2, hud + oy * DYN_UNIT + DYN_UNIT // 2
            r = int(DYN_UNIT * 0.36)
            pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
            pygame.draw.polygon(self._screen, OBSTACLE_COLOR, pts)

        # --- Agent (flash red briefly after hitting an obstacle) -----------
        ax, ay = self.agent
        size = int(DYN_UNIT * 0.65)
        cx, cy = ax * DYN_UNIT + DYN_UNIT // 2, hud + ay * DYN_UNIT + DYN_UNIT // 2
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        color = OBSTACLE_COLOR if self._hit_flash > 0 else AGENT_COLOR
        pygame.draw.rect(self._screen, color, rect)

        # --- "-1" popup over the agent during the flash --------------------
        if self._hit_flash > 0:
            popup = self._popup_font.render("-1", True, OBSTACLE_COLOR)
            self._screen.blit(popup, popup.get_rect(center=(cx, cy - size // 2 - 14)))
            self._hit_flash -= 1

        pygame.display.flip()
        time.sleep(FPS_DELAY)
