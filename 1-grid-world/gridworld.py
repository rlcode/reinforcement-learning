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


# ---------------------------------------------------------------------------
# Policy / Value Iteration support (Dynamic Programming).
# ---------------------------------------------------------------------------

# Action indices used by the DP algorithms.
#   0: up    1: down    2: left    3: right
# Coordinate convention: state is [x, y] where x is the *row* (0 = top) and
# y is the *column* (0 = left).  This matches the original repo's code.
DP_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class PolicyEnv:
    """Pure-data env for policy/value iteration.

    No rendering, no time, no agent — just the MDP: states, transitions,
    rewards.  Layout matches the original tkinter-based code:
        Goal      at (2, 2): reward +1, terminal
        Obstacles at (1, 2) and (2, 1): reward -1
    """

    transition_probability = 1
    possible_actions = [0, 1, 2, 3]

    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[0.0] * self.width for _ in range(self.height)]
        self.reward[2][2] = 1.0
        self.reward[1][2] = -1.0
        self.reward[2][1] = -1.0
        # Useful to the agents:
        self.all_state = [[x, y] for x in range(self.width) for y in range(self.height)]

    def get_all_states(self):
        return self.all_state

    def state_after_action(self, state, action_index):
        dx, dy = DP_ACTIONS[action_index]
        return self._check_boundary([state[0] + dx, state[1] + dy])

    @staticmethod
    def _check_boundary(state):
        state[0] = max(0, min(WIDTH - 1, state[0]))
        state[1] = max(0, min(HEIGHT - 1, state[1]))
        return state

    def get_reward(self, state, action):
        ns = self.state_after_action(state, action)
        return self.reward[ns[0]][ns[1]]

    def get_transition_prob(self, state, action):
        return self.transition_probability


class GraphicDisplay:
    """Pygame button-driven viewer for policy / value iteration.

    Visual elements:
      - 5x5 grid (100 px cells), goal/obstacle markers, agent square.
      - Reward labels ("R : 1.0" / "R : -1.0") in the relevant cells.
      - Optional V(s) text in each cell (set via show_values).
      - Optional policy arrows in each cell (set via show_arrows).
      - A row of 4 buttons at the bottom; the caller passes
        (label, handler) tuples.

    Click handling: the mainloop polls pygame events; clicking a button's
    rect calls the corresponding handler.
    """

    BUTTON_BAR_HEIGHT = 50

    def __init__(self, agent, title, buttons=None):
        """`buttons` is a list of (label, callable) tuples (1-4 entries).
        Can also be assigned after construction (handlers commonly need to
        capture the display itself, which is awkward in a single expression)."""
        self.agent = agent
        self.env = PolicyEnv()
        self.title = title
        self.buttons = buttons or []
        # Agent grid position for the "Move" animation.
        self.agent_pos = [0, 0]
        # Display state.
        self._screen = None
        self._font = None
        self._small_font = None
        self._value_table = None     # 2-D list to overlay, or None
        # Policy overlay: per-cell list of (action_index_set,) or weighted probs.
        self._policy_arrows = None   # policy_table[row][col] = [p_up, p_down, p_left, p_right]

    # ---- public API used by the algorithm's button handlers --------------

    def show_values(self, value_table):
        """Overlay a V(s) table (value_table[row][col]) on the grid."""
        self._value_table = value_table

    def show_arrows(self, policy_table):
        """Overlay policy arrows.  policy_table[row][col] is a 4-list of
        probabilities (up, down, left, right); any positive entry draws an
        arrow in that direction."""
        self._policy_arrows = policy_table

    def clear(self):
        self._value_table = None
        self._policy_arrows = None

    def move_along_policy(self, action_picker):
        """Animate the agent moving along the greedy policy.

        `action_picker(state) -> int | list[int] | None`:
          - int:        the action to take
          - list[int]:  pick one (used by value-iteration's tied actions)
          - None/[]:    stop (terminal state reached)
        """
        self.agent_pos = [0, 0]
        while True:
            self._render()
            pygame.time.wait(200)
            action = action_picker(list(self.agent_pos))
            if action is None or action == [] or action == 0.0:
                break
            if isinstance(action, list):
                if not action:
                    break
                action = action[0]  # tie-break: pick the first
            dx, dy = DP_ACTIONS[action]
            self.agent_pos = [
                max(0, min(WIDTH - 1, self.agent_pos[0] + dx)),
                max(0, min(HEIGHT - 1, self.agent_pos[1] + dy)),
            ]

    # ---- mainloop --------------------------------------------------------

    def mainloop(self):
        self._ensure_display()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._dispatch_click(event.pos)
            self._render()
            time.sleep(0.03)
        pygame.quit()

    # ---- internals -------------------------------------------------------

    def _ensure_display(self):
        if self._screen is not None:
            return
        pygame.init()
        pygame.display.set_caption(self.title)
        height = HEIGHT * UNIT + self.BUTTON_BAR_HEIGHT
        self._screen = pygame.display.set_mode((WIDTH * UNIT, height))
        self._font = pygame.font.SysFont(None, 22)
        self._small_font = pygame.font.SysFont(None, 16)

    def _button_rects(self):
        bar_y = HEIGHT * UNIT + 8
        bar_h = self.BUTTON_BAR_HEIGHT - 16
        slot_w = (WIDTH * UNIT) // max(len(self.buttons), 1)
        rects = []
        for i, _ in enumerate(self.buttons):
            r = pygame.Rect(i * slot_w + 8, bar_y, slot_w - 16, bar_h)
            rects.append(r)
        return rects

    def _dispatch_click(self, pos):
        for rect, (label, handler) in zip(self._button_rects(), self.buttons):
            if rect.collidepoint(pos):
                handler()
                return

    def _render(self):
        self._screen.fill(WHITE)
        self._draw_grid()
        self._draw_reward_cells()
        self._draw_value_overlay()
        self._draw_policy_arrows()
        self._draw_agent()
        self._draw_buttons()
        pygame.display.flip()

    def _draw_grid(self):
        for c in range(WIDTH + 1):
            pygame.draw.line(self._screen, GRID_LINE,
                             (c * UNIT, 0), (c * UNIT, HEIGHT * UNIT))
        for r in range(HEIGHT + 1):
            pygame.draw.line(self._screen, GRID_LINE,
                             (0, r * UNIT), (WIDTH * UNIT, r * UNIT))

    def _cell_center(self, row, col):
        # Convention: state = [row, col]; on screen, col -> x, row -> y.
        return (col * UNIT + UNIT // 2, row * UNIT + UNIT // 2)

    def _draw_reward_cells(self):
        # Triangles at (1,2) and (2,1); circle at (2,2).
        cx, cy = self._cell_center(2, 2)
        pygame.draw.circle(self._screen, GOAL_COLOR, (cx, cy), int(UNIT * 0.33))
        for rr, cc in [(1, 2), (2, 1)]:
            cx, cy = self._cell_center(rr, cc)
            r = int(UNIT * 0.36)
            pygame.draw.polygon(self._screen, OBSTACLE_COLOR,
                                [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)])
        # "R : ±1.0" reward labels (top-left corner of each cell).
        for rr, cc, txt in [(2, 2, "R : +1.0"), (1, 2, "R : -1.0"), (2, 1, "R : -1.0")]:
            x = cc * UNIT + 6
            y = rr * UNIT + 4
            surf = self._small_font.render(txt, True, TEXT_COLOR)
            self._screen.blit(surf, (x, y))

    def _draw_value_overlay(self):
        if self._value_table is None:
            return
        for r in range(HEIGHT):
            for c in range(WIDTH):
                v = self._value_table[r][c]
                surf = self._font.render(f"{v:.2f}", True, TEXT_COLOR)
                cx, cy = self._cell_center(r, c)
                self._screen.blit(surf, surf.get_rect(center=(cx, cy + UNIT // 4)))

    def _draw_policy_arrows(self):
        if self._policy_arrows is None:
            return
        for r in range(HEIGHT):
            for c in range(WIDTH):
                probs = self._policy_arrows[r][c]
                if not probs:
                    continue
                cx, cy = self._cell_center(r, c)
                # Up, Down, Left, Right offsets (point toward cell edge).
                arrows = [(0, -UNIT * 0.32), (0, UNIT * 0.32),
                          (-UNIT * 0.32, 0), (UNIT * 0.32, 0)]
                for i, p in enumerate(probs):
                    if p > 0:
                        self._draw_arrow(cx, cy, cx + arrows[i][0], cy + arrows[i][1])

    def _draw_arrow(self, x0, y0, x1, y1):
        pygame.draw.line(self._screen, BLACK, (x0, y0), (x1, y1), 2)
        # Tiny arrowhead: short perpendicular segments.
        import math
        ang = math.atan2(y1 - y0, x1 - x0)
        head = 6
        for sign in (-1, 1):
            a = ang + sign * 2.5
            pygame.draw.line(self._screen, BLACK, (x1, y1),
                             (x1 - head * math.cos(a), y1 - head * math.sin(a)), 2)

    def _draw_agent(self):
        size = int(UNIT * 0.55)
        cx, cy = self._cell_center(self.agent_pos[0], self.agent_pos[1])
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        # Hollow blue outline so the V/arrows underneath stay visible.
        pygame.draw.rect(self._screen, AGENT_COLOR, rect, 3)

    def _draw_buttons(self):
        for rect, (label, _) in zip(self._button_rects(), self.buttons):
            pygame.draw.rect(self._screen, (220, 220, 220), rect)
            pygame.draw.rect(self._screen, BLACK, rect, 1)
            surf = self._font.render(label, True, BLACK)
            self._screen.blit(surf, surf.get_rect(center=rect.center))
