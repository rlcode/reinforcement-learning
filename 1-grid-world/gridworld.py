"""Shared pygame grid-world environments and DP viewer.

Cross-platform replacement for the original tkinter envs.  Three pieces:

  - Env         : static 5x5 grid (SARSA, Q-learning).
  - DynamicEnv  : 5x5 grid with horizontally-moving obstacles
                  (Deep SARSA, REINFORCE).
  - PolicyEnv   : pure-data MDP (policy / value iteration).
  - GraphicDisplay : pygame button-driven viewer for the DP algorithms.
"""
import math
import time

import pygame

UNIT = 100        # cell size for static Env and the DP viewer
DYN_UNIT = 50     # cell size for DynamicEnv
WIDTH = 5
HEIGHT = 5
FPS_DELAY = 0.03

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_LINE = (200, 200, 200)
AGENT_COLOR = (60, 120, 220)
OBSTACLE_COLOR = (220, 60, 60)
GOAL_COLOR = (60, 200, 100)
TEXT_COLOR = (40, 40, 40)


# ---------------------------------------------------------------------------
# Shared pygame helpers (module-level so each class can mix and match).
# ---------------------------------------------------------------------------

def pump_events():
    """Drain the event queue; raise SystemExit if the window was closed."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit


def draw_grid(surface, unit, y_off=0):
    for c in range(WIDTH + 1):
        pygame.draw.line(surface, GRID_LINE,
                         (c * unit, y_off), (c * unit, y_off + HEIGHT * unit))
    for r in range(HEIGHT + 1):
        pygame.draw.line(surface, GRID_LINE,
                         (0, y_off + r * unit), (WIDTH * unit, y_off + r * unit))


def cell_center(x, y, unit, y_off=0):
    return (x * unit + unit // 2, y_off + y * unit + unit // 2)


def draw_square(surface, x, y, unit, color, y_off=0, filled=True):
    size = int(unit * 0.65)
    cx, cy = cell_center(x, y, unit, y_off)
    rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
    pygame.draw.rect(surface, color, rect, 0 if filled else 3)
    return rect


def draw_circle(surface, x, y, unit, color, y_off=0):
    pygame.draw.circle(surface, color, cell_center(x, y, unit, y_off), int(unit * 0.33))


def draw_triangle(surface, x, y, unit, color, y_off=0):
    cx, cy = cell_center(x, y, unit, y_off)
    r = int(unit * 0.36)
    pygame.draw.polygon(surface, color,
                        [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)])


# ---------------------------------------------------------------------------
# Static grid (tabular SARSA, Q-learning).
# ---------------------------------------------------------------------------

class Env:
    """Static-obstacle 5x5 grid-world.

    Agent starts at (0,0).  Obstacles at (1,2) and (2,1) (-100, terminal).
    Goal at (2,2) (+100, terminal).  Actions: 0=up, 1=down, 2=left, 3=right.
    """

    n_actions = 4

    # Q-overlay text offsets per action (up/down/left/right) relative to cell center.
    _Q_OFFSETS = [(0, -UNIT // 2 + 10), (0, UNIT // 2 - 10),
                  (-UNIT // 2 + 15, 0), (UNIT // 2 - 15, 0)]

    def __init__(self, title="GridWorld"):
        self.title = title
        self.agent = [0, 0]
        self.obstacles = [[1, 2], [2, 1]]
        self.goal = [2, 2]
        self._screen = None
        self._font = None
        self._q_overlay = None

    # ---- RL API ----------------------------------------------------------

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

    def print_value_all(self, q_table):
        """Push a Q-table; drawn on the next render() call."""
        self._q_overlay = q_table

    # ---- rendering -------------------------------------------------------

    def render(self):
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption(self.title)
            self._screen = pygame.display.set_mode((WIDTH * UNIT, HEIGHT * UNIT))
            self._font = pygame.font.SysFont(None, 18)
        pump_events()

        s = self._screen
        s.fill(WHITE)
        draw_grid(s, UNIT)
        for ox, oy in self.obstacles:
            draw_triangle(s, ox, oy, UNIT, OBSTACLE_COLOR)
        draw_circle(s, *self.goal, unit=UNIT, color=GOAL_COLOR)
        draw_square(s, *self.agent, unit=UNIT, color=AGENT_COLOR)
        if self._q_overlay is not None:
            self._draw_q_overlay(self._q_overlay)

        pygame.display.flip()
        time.sleep(FPS_DELAY)

    def _draw_q_overlay(self, q_table):
        # q_table: dict keyed by str([x, y]) -> [up, down, left, right] values.
        for x in range(WIDTH):
            for y in range(HEIGHT):
                qs = q_table.get(str([x, y]))
                if qs is None:
                    continue
                cx, cy = cell_center(x, y, UNIT)
                for i, q in enumerate(qs):
                    text = self._font.render(f"{q:+.2f}", True, TEXT_COLOR)
                    dx, dy = self._Q_OFFSETS[i]
                    self._screen.blit(text, text.get_rect(center=(cx + dx, cy + dy)))


# ---------------------------------------------------------------------------
# Dynamic grid (Deep SARSA, REINFORCE) — moving obstacles, HUD, hit flash.
# ---------------------------------------------------------------------------

class DynamicEnv:
    """5x5 grid with 3 obstacles that jump left/right one cell every 2nd step.

    Goal at (4,4); only the goal terminates.  Hitting an obstacle costs -1
    and continues.  `step_penalty` adds a per-step cost (REINFORCE uses 0.1
    to encourage shorter paths).  State is a 15-dim relative encoding:
    for each obstacle [dx, dy, -1, direction] (4 dims) and for the goal
    [dx, dy, +1] (3 dims).  Actions: 0=up, 1=down, 2=right, 3=left.
    """

    n_actions = 4
    state_size = 15
    HUD_HEIGHT = 32

    def __init__(self, title="DynamicGridWorld", step_penalty=0.0, render_mode="human"):
        self.title = title
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        self.agent = [0, 0]
        self.obstacles_init = [[0, 1], [1, 2], [2, 3]]
        self.goal = [4, 4]
        self.obstacles = []
        self.counter = 0

        # HUD state.
        self.episode = 0
        self.score = 0.0
        self._hit_flash = 0  # remaining frames to flash the agent red

        self._screen = None
        self._hud_font = None
        self._popup_font = None

    # ---- RL API ----------------------------------------------------------

    def reset(self):
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
        # Render every step (matches the original env's behavior).
        self.render()
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
        if reward < -self.step_penalty:
            self._hit_flash = 4
        return self._get_state(), reward, done

    def _reward_and_done(self):
        if self.agent == self.goal:
            return 1.0, True
        # Sums if the agent lands on multiple obstacles in the same cell.
        r = sum(-1.0 for o in self.obstacles if o["state"] == self.agent)
        return r, False

    def _move_obstacles(self):
        for obs in self.obstacles:
            x, _ = obs["state"]
            # Bounce at the edges. direction=-1 -> right, +1 -> left.
            if x == WIDTH - 1:
                obs["direction"] = 1
            elif x == 0:
                obs["direction"] = -1
            obs["state"][0] += 1 if obs["direction"] == -1 else -1

    def _get_state(self):
        ax, ay = self.agent
        s = []
        for obs in self.obstacles:
            ox, oy = obs["state"]
            s += [ox - ax, oy - ay, -1, obs["direction"]]
        gx, gy = self.goal
        s += [gx - ax, gy - ay, 1]
        return s

    # ---- rendering -------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption(self.title)
            self._screen = pygame.display.set_mode(
                (WIDTH * DYN_UNIT, HEIGHT * DYN_UNIT + self.HUD_HEIGHT))
            self._hud_font = pygame.font.SysFont(None, 22)
            self._popup_font = pygame.font.SysFont(None, 28)
        pump_events()

        s = self._screen
        hud = self.HUD_HEIGHT
        s.fill(WHITE)

        # HUD bar.
        pygame.draw.rect(s, (30, 30, 30), pygame.Rect(0, 0, WIDTH * DYN_UNIT, hud))
        text = self._hud_font.render(
            f"Episode: {self.episode}    Score: {self.score:+.1f}", True, (240, 240, 240))
        s.blit(text, (8, (hud - text.get_height()) // 2))

        # Grid + landmarks.
        draw_grid(s, DYN_UNIT, y_off=hud)
        draw_circle(s, *self.goal, unit=DYN_UNIT, color=GOAL_COLOR, y_off=hud)
        for obs in self.obstacles:
            draw_triangle(s, *obs["state"], unit=DYN_UNIT, color=OBSTACLE_COLOR, y_off=hud)

        # Agent (flash red briefly after an obstacle hit).
        color = OBSTACLE_COLOR if self._hit_flash > 0 else AGENT_COLOR
        agent_rect = draw_square(s, *self.agent, unit=DYN_UNIT, color=color, y_off=hud)
        if self._hit_flash > 0:
            popup = self._popup_font.render("-1", True, OBSTACLE_COLOR)
            s.blit(popup, popup.get_rect(center=(agent_rect.centerx, agent_rect.top - 14)))
            self._hit_flash -= 1

        pygame.display.flip()
        time.sleep(FPS_DELAY)


# ---------------------------------------------------------------------------
# Policy / Value Iteration support (Dynamic Programming).
# ---------------------------------------------------------------------------

# 0: up, 1: down, 2: left, 3: right.  state = [row, col] (matches original).
DP_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class PolicyEnv:
    """Pure-data MDP for policy/value iteration: states, transitions, rewards.

    Goal at (2,2): +1, terminal.  Obstacles at (1,2) and (2,1): -1.
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
        self.all_state = [[x, y] for x in range(self.width) for y in range(self.height)]

    def get_all_states(self):
        return self.all_state

    def state_after_action(self, state, action_index):
        dx, dy = DP_ACTIONS[action_index]
        nx = max(0, min(WIDTH - 1, state[0] + dx))
        ny = max(0, min(HEIGHT - 1, state[1] + dy))
        return [nx, ny]

    def get_reward(self, state, action):
        ns = self.state_after_action(state, action)
        return self.reward[ns[0]][ns[1]]

    def get_transition_prob(self, state, action):
        return self.transition_probability


class GraphicDisplay:
    """Pygame button-driven viewer for policy / value iteration.

    Buttons are a list of (label, handler) tuples; clicking dispatches to
    the matching handler.  show_values overlays a V(s) table, show_arrows
    overlays a policy table, move_along_policy animates the agent along
    greedy actions.
    """

    BUTTON_BAR_HEIGHT = 50

    def __init__(self, agent, title, buttons=None):
        self.agent = agent
        self.env = PolicyEnv()
        self.title = title
        # Assignable after construction so handlers can close over `display`.
        self.buttons = buttons or []
        self.agent_pos = [0, 0]
        self._screen = None
        self._font = None
        self._small_font = None
        self._value_table = None      # 2-D list of V(s), or None
        self._policy_arrows = None    # 2-D list of [p_up, p_down, p_left, p_right]

    # ---- public API (called by the button handlers) ----------------------

    def show_values(self, value_table):
        self._value_table = value_table

    def show_arrows(self, policy_table):
        self._policy_arrows = policy_table

    def clear(self):
        self._value_table = None
        self._policy_arrows = None

    def move_along_policy(self, action_picker):
        """Animate the agent under the greedy policy.

        `action_picker(state)` may return int (single action), list[int]
        (tied actions; first is taken), or None/[] (terminal — stop).
        """
        self.agent_pos = [0, 0]
        while True:
            self._render()
            pygame.time.wait(200)
            action = action_picker(list(self.agent_pos))
            if action is None or action == [] or action == 0.0:
                break
            if isinstance(action, list):
                action = action[0]
            dx, dy = DP_ACTIONS[action]
            self.agent_pos = [max(0, min(WIDTH - 1, self.agent_pos[0] + dx)),
                              max(0, min(HEIGHT - 1, self.agent_pos[1] + dy))]

    def mainloop(self):
        self._ensure_display()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._dispatch_click(event.pos)
            self._render()
            time.sleep(0.03)

    # ---- internals -------------------------------------------------------

    def _ensure_display(self):
        if self._screen is not None:
            return
        pygame.init()
        pygame.display.set_caption(self.title)
        h = HEIGHT * UNIT + self.BUTTON_BAR_HEIGHT
        self._screen = pygame.display.set_mode((WIDTH * UNIT, h))
        self._font = pygame.font.SysFont(None, 22)
        self._small_font = pygame.font.SysFont(None, 16)

    def _button_rects(self):
        bar_y = HEIGHT * UNIT + 8
        bar_h = self.BUTTON_BAR_HEIGHT - 16
        slot_w = (WIDTH * UNIT) // max(len(self.buttons), 1)
        return [pygame.Rect(i * slot_w + 8, bar_y, slot_w - 16, bar_h)
                for i in range(len(self.buttons))]

    def _dispatch_click(self, pos):
        for rect, (_, handler) in zip(self._button_rects(), self.buttons):
            if rect.collidepoint(pos):
                handler()
                return

    def _render(self):
        s = self._screen
        s.fill(WHITE)
        draw_grid(s, UNIT)

        # Goal + obstacles + their "R : ±1.0" labels.
        draw_circle(s, 2, 2, UNIT, GOAL_COLOR)
        for x, y in [(1, 2), (2, 1)]:
            draw_triangle(s, y, x, UNIT, OBSTACLE_COLOR)  # row -> y, col -> x
        for row, col, txt in [(2, 2, "R : +1.0"), (1, 2, "R : -1.0"), (2, 1, "R : -1.0")]:
            label = self._small_font.render(txt, True, TEXT_COLOR)
            s.blit(label, (col * UNIT + 6, row * UNIT + 4))

        self._draw_value_overlay()
        self._draw_policy_arrows()

        # Agent: hollow outline so V/arrows stay visible.
        ax, ay = self.agent_pos
        size = int(UNIT * 0.55)
        cx, cy = ay * UNIT + UNIT // 2, ax * UNIT + UNIT // 2
        pygame.draw.rect(s, AGENT_COLOR,
                         pygame.Rect(cx - size // 2, cy - size // 2, size, size), 3)

        self._draw_buttons()
        pygame.display.flip()

    def _draw_value_overlay(self):
        if self._value_table is None:
            return
        for r in range(HEIGHT):
            for c in range(WIDTH):
                surf = self._font.render(f"{self._value_table[r][c]:.2f}", True, TEXT_COLOR)
                cx, cy = c * UNIT + UNIT // 2, r * UNIT + UNIT // 2
                self._screen.blit(surf, surf.get_rect(center=(cx, cy + UNIT // 4)))

    def _draw_policy_arrows(self):
        if self._policy_arrows is None:
            return
        # Action -> (dx, dy) offset from cell center toward an edge.
        edge = [(0, -UNIT * 0.32), (0, UNIT * 0.32),
                (-UNIT * 0.32, 0), (UNIT * 0.32, 0)]
        for r in range(HEIGHT):
            for c in range(WIDTH):
                probs = self._policy_arrows[r][c]
                if not probs:
                    continue
                cx, cy = c * UNIT + UNIT // 2, r * UNIT + UNIT // 2
                for i, p in enumerate(probs):
                    if p > 0:
                        self._draw_arrow(cx, cy, cx + edge[i][0], cy + edge[i][1])

    def _draw_arrow(self, x0, y0, x1, y1):
        pygame.draw.line(self._screen, BLACK, (x0, y0), (x1, y1), 2)
        ang = math.atan2(y1 - y0, x1 - x0)
        head = 6
        for sign in (-1, 1):
            a = ang + sign * 2.5
            pygame.draw.line(self._screen, BLACK, (x1, y1),
                             (x1 - head * math.cos(a), y1 - head * math.sin(a)), 2)

    def _draw_buttons(self):
        for rect, (label, _) in zip(self._button_rects(), self.buttons):
            pygame.draw.rect(self._screen, (220, 220, 220), rect)
            pygame.draw.rect(self._screen, BLACK, rect, 1)
            surf = self._font.render(label, True, BLACK)
            self._screen.blit(surf, surf.get_rect(center=rect.center))
