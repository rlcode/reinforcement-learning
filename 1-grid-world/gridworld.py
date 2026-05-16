"""Pygame grid-world envs and DP viewer (shared by all six algorithms)."""
import math
import time

import pygame

UNIT, DYN_UNIT, WIDTH, HEIGHT, FPS_DELAY = 100, 50, 5, 5, 0.03
WHITE, BLACK, GRID_LINE = (255, 255, 255), (0, 0, 0), (200, 200, 200)
AGENT_COLOR, OBSTACLE_COLOR, GOAL_COLOR = (60, 120, 220), (220, 60, 60), (60, 200, 100)
TEXT_COLOR = (40, 40, 40)

# 0=up, 1=down, 2=left, 3=right for DP / static Env (state = [row, col] for DP,
# [col, row] for Env — see each class). DynamicEnv uses 0=up,1=down,2=right,3=left.
DP_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _pump_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit


def _open(title, size):
    pygame.init()
    pygame.display.set_caption(title)
    return pygame.display.set_mode(size)


def _grid_lines(surf, unit, y_off=0):
    for c in range(WIDTH + 1):
        pygame.draw.line(surf, GRID_LINE, (c * unit, y_off), (c * unit, y_off + HEIGHT * unit))
    for r in range(HEIGHT + 1):
        pygame.draw.line(surf, GRID_LINE, (0, y_off + r * unit), (WIDTH * unit, y_off + r * unit))


def _center(x, y, unit, y_off=0):
    return x * unit + unit // 2, y_off + y * unit + unit // 2


def _square(surf, x, y, unit, color, y_off=0, fill=True):
    cx, cy = _center(x, y, unit, y_off)
    s = int(unit * 0.65)
    rect = pygame.Rect(cx - s // 2, cy - s // 2, s, s)
    pygame.draw.rect(surf, color, rect, 0 if fill else 3)
    return rect


def _circle(surf, x, y, unit, color, y_off=0):
    pygame.draw.circle(surf, color, _center(x, y, unit, y_off), int(unit * 0.33))


def _triangle(surf, x, y, unit, color, y_off=0):
    cx, cy = _center(x, y, unit, y_off)
    r = int(unit * 0.36)
    pygame.draw.polygon(surf, color, [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)])


# ---------------------------------------------------------------------------
class Env:
    """Static 5x5 grid for tabular SARSA / Q-learning.  agent=[col,row]."""
    n_actions = 4

    def __init__(self, title="GridWorld"):
        self.title = title
        self.agent = [0, 0]
        self.obstacles = [[1, 2], [2, 1]]
        self.goal = [2, 2]
        self.q_overlay = None  # set by print_value_all; rendered on next render()
        self._screen = None

    def reset(self):
        self.agent = [0, 0]
        if self._screen is not None:
            self.render()
            time.sleep(0.3)
        return list(self.agent)

    def step(self, action):
        x, y = self.agent
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < HEIGHT - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < WIDTH - 1: x += 1
        self.agent = [x, y]
        if self.agent == self.goal: return list(self.agent), 100, True
        if self.agent in self.obstacles: return list(self.agent), -100, True
        return list(self.agent), 0, False

    def print_value_all(self, q_table):
        self.q_overlay = q_table

    def render(self):
        if self._screen is None:
            self._screen = _open(self.title, (WIDTH * UNIT, HEIGHT * UNIT))
            self._font = pygame.font.SysFont(None, 18)
        _pump_events()
        s = self._screen
        s.fill(WHITE)
        _grid_lines(s, UNIT)
        for ox, oy in self.obstacles:
            _triangle(s, ox, oy, UNIT, OBSTACLE_COLOR)
        _circle(s, *self.goal, unit=UNIT, color=GOAL_COLOR)
        _square(s, *self.agent, unit=UNIT, color=AGENT_COLOR)
        if self.q_overlay is not None:
            offsets = [(0, -UNIT // 2 + 10), (0, UNIT // 2 - 10),
                       (-UNIT // 2 + 15, 0), (UNIT // 2 - 15, 0)]
            for x in range(WIDTH):
                for y in range(HEIGHT):
                    qs = self.q_overlay.get(str([x, y]))
                    if qs is None: continue
                    cx, cy = _center(x, y, UNIT)
                    for i, q in enumerate(qs):
                        t = self._font.render(f"{q:+.2f}", True, TEXT_COLOR)
                        s.blit(t, t.get_rect(center=(cx + offsets[i][0], cy + offsets[i][1])))
        pygame.display.flip()
        time.sleep(FPS_DELAY)


# ---------------------------------------------------------------------------
class DynamicEnv:
    """5x5 grid with horizontally-bouncing obstacles (Deep SARSA, REINFORCE).

    Goal at (4,4) terminates; obstacle hit costs -1 but continues.
    State: 15-dim relative encoding (4 per obstacle, 3 for goal).
    Actions: 0=up, 1=down, 2=right, 3=left (note: differs from Env).
    """
    n_actions = 4
    state_size = 15
    HUD = 32

    def __init__(self, title="DynamicGridWorld", step_penalty=0.0, render_mode="human"):
        self.title, self.step_penalty, self.render_mode = title, step_penalty, render_mode
        self.agent = [0, 0]
        self.obstacles_init = [[0, 1], [1, 2], [2, 3]]
        self.goal = [4, 4]
        self.obstacles = []
        self.counter, self.episode, self.score, self._hit = 0, 0, 0.0, 0
        self._screen = None

    def reset(self):
        if self.counter > 0 or self.score != 0.0:
            self.episode += 1
        self.agent, self.counter, self.score, self._hit = [0, 0], 0, 0.0, 0
        self.obstacles = [{"state": list(p), "direction": -1} for p in self.obstacles_init]
        if self._screen is not None:
            self.render(); time.sleep(0.3)
        return self._state()

    def step(self, action):
        self.counter += 1
        self.render()
        if self.counter % 2 == 1:
            for o in self.obstacles:
                if o["state"][0] == WIDTH - 1: o["direction"] = 1
                elif o["state"][0] == 0: o["direction"] = -1
                o["state"][0] += 1 if o["direction"] == -1 else -1

        x, y = self.agent
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < HEIGHT - 1: y += 1
        elif action == 2 and x < WIDTH - 1: x += 1
        elif action == 3 and x > 0: x -= 1
        self.agent = [x, y]

        done = self.agent == self.goal
        reward = 1.0 if done else sum(-1.0 for o in self.obstacles if o["state"] == self.agent)
        reward -= self.step_penalty
        self.score += reward
        if reward < -self.step_penalty:
            self._hit = 4
        return self._state(), reward, done

    def _state(self):
        ax, ay = self.agent
        s = []
        for o in self.obstacles:
            ox, oy = o["state"]
            s += [ox - ax, oy - ay, -1, o["direction"]]
        s += [self.goal[0] - ax, self.goal[1] - ay, 1]
        return s

    def render(self):
        if self.render_mode is None: return
        if self._screen is None:
            self._screen = _open(self.title, (WIDTH * DYN_UNIT, HEIGHT * DYN_UNIT + self.HUD))
            self._hud_font = pygame.font.SysFont(None, 22)
            self._popup_font = pygame.font.SysFont(None, 28)
        _pump_events()
        s, hud = self._screen, self.HUD
        s.fill(WHITE)
        pygame.draw.rect(s, (30, 30, 30), pygame.Rect(0, 0, WIDTH * DYN_UNIT, hud))
        t = self._hud_font.render(f"Episode: {self.episode}    Score: {self.score:+.1f}",
                                  True, (240, 240, 240))
        s.blit(t, (8, (hud - t.get_height()) // 2))
        _grid_lines(s, DYN_UNIT, y_off=hud)
        _circle(s, *self.goal, unit=DYN_UNIT, color=GOAL_COLOR, y_off=hud)
        for o in self.obstacles:
            _triangle(s, *o["state"], unit=DYN_UNIT, color=OBSTACLE_COLOR, y_off=hud)
        rect = _square(s, *self.agent, unit=DYN_UNIT,
                       color=OBSTACLE_COLOR if self._hit > 0 else AGENT_COLOR, y_off=hud)
        if self._hit > 0:
            p = self._popup_font.render("-1", True, OBSTACLE_COLOR)
            s.blit(p, p.get_rect(center=(rect.centerx, rect.top - 14)))
            self._hit -= 1
        pygame.display.flip()
        time.sleep(FPS_DELAY)


# ---------------------------------------------------------------------------
class PolicyEnv:
    """Pure-data MDP for policy/value iteration.  state = [row, col]."""
    transition_probability = 1
    possible_actions = [0, 1, 2, 3]

    def __init__(self):
        self.width, self.height = WIDTH, HEIGHT
        self.reward = [[0.0] * WIDTH for _ in range(HEIGHT)]
        self.reward[2][2], self.reward[1][2], self.reward[2][1] = 1.0, -1.0, -1.0
        self.all_state = [[x, y] for x in range(WIDTH) for y in range(HEIGHT)]

    def get_all_states(self):
        return self.all_state

    def state_after_action(self, state, action):
        dx, dy = DP_ACTIONS[action]
        return [max(0, min(WIDTH - 1, state[0] + dx)), max(0, min(HEIGHT - 1, state[1] + dy))]

    def get_reward(self, state, action):
        ns = self.state_after_action(state, action)
        return self.reward[ns[0]][ns[1]]

    def get_transition_prob(self, state, action):
        return self.transition_probability


# ---------------------------------------------------------------------------
class GraphicDisplay:
    """Pygame button-driven viewer for policy / value iteration.

    Set `display.buttons = [(label, handler[, enabled]), ...]` (up to 4).
    `enabled` is an optional zero-arg callable returning bool; when it
    returns False the button is greyed out and clicks are ignored.
    `show_values(V)` / `show_arrows(policy_table)` overlay; `clear()`
    removes them. `move_along_policy(picker)` animates greedy moves.
    """
    BAR = 50

    def __init__(self, agent, title, buttons=None):
        self.agent = agent
        self.env = PolicyEnv()
        self.title = title
        self.buttons = buttons or []
        self.agent_pos = [0, 0]
        # Per-label click counts, available to button `enabled` predicates.
        self.clicks = {}
        # Brief "pressed" flash so clicks feel responsive.
        self._press_label = None
        self._press_frames = 0
        self._screen = None
        self._values = None
        self._arrows = None

    def click_count(self, label):
        return self.clicks.get(label, 0)

    def show_values(self, v): self._values = v
    def show_arrows(self, p): self._arrows = p
    def clear(self): self._values = self._arrows = None

    def move_along_policy(self, picker):
        self.agent_pos = [0, 0]
        while True:
            self._render(); pygame.time.wait(200)
            r, c = self.agent_pos
            # Stop at the goal cell — picker may not be defined there
            # (policy iteration's get_action crashes on the terminal state).
            if self.env.reward[r][c] > 0:
                break
            a = picker(list(self.agent_pos))
            if a is None or a == [] or a == 0.0: break
            if isinstance(a, list): a = a[0]
            dx, dy = DP_ACTIONS[a]
            self.agent_pos = [max(0, min(WIDTH - 1, self.agent_pos[0] + dx)),
                              max(0, min(HEIGHT - 1, self.agent_pos[1] + dy))]

    def mainloop(self):
        self._screen = _open(self.title, (WIDTH * UNIT, HEIGHT * UNIT + self.BAR))
        self._font = pygame.font.SysFont(None, 22)
        self._small = pygame.font.SysFont(None, 16)
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); return
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for rect, btn in zip(self._btn_rects(), self.buttons):
                        if rect.collidepoint(e.pos) and self._btn_enabled(btn):
                            label = btn[0]
                            self._press_label = label
                            self._press_frames = 6
                            self._render()  # draw the pressed state immediately
                            btn[1]()
                            self.clicks[label] = self.clicks.get(label, 0) + 1
                            break
            self._render()
            time.sleep(0.03)

    @staticmethod
    def _btn_enabled(btn):
        # Button tuple is (label, handler) or (label, handler, enabled_fn).
        return btn[2]() if len(btn) >= 3 else True

    def _btn_rects(self):
        n = max(len(self.buttons), 1)
        w = (WIDTH * UNIT) // n
        return [pygame.Rect(i * w + 8, HEIGHT * UNIT + 8, w - 16, self.BAR - 16)
                for i in range(len(self.buttons))]

    def _render(self):
        s = self._screen
        s.fill(WHITE)
        _grid_lines(s, UNIT)
        # PolicyEnv state is [row, col] but our draw helpers take [col, row] — swap.
        _circle(s, 2, 2, UNIT, GOAL_COLOR)
        for row, col in [(1, 2), (2, 1)]:
            _triangle(s, col, row, UNIT, OBSTACLE_COLOR)
            label = self._small.render("R : -1.0", True, TEXT_COLOR)
            s.blit(label, (col * UNIT + 6, row * UNIT + 4))
        s.blit(self._small.render("R : +1.0", True, TEXT_COLOR), (2 * UNIT + 6, 2 * UNIT + 4))

        # Agent first (filled), so V text and arrows render on top of it.
        r, c = self.agent_pos
        sz = int(UNIT * 0.55)
        cx, cy = c * UNIT + UNIT // 2, r * UNIT + UNIT // 2
        pygame.draw.rect(s, AGENT_COLOR, pygame.Rect(cx - sz // 2, cy - sz // 2, sz, sz))

        if self._values is not None:
            for r in range(HEIGHT):
                for c in range(WIDTH):
                    t = self._font.render(f"{self._values[r][c]:.2f}", True, TEXT_COLOR)
                    cx, cy = c * UNIT + UNIT // 2, r * UNIT + UNIT // 2
                    s.blit(t, t.get_rect(center=(cx, cy + UNIT // 4)))

        if self._arrows is not None:
            edge = [(0, -UNIT * 0.32), (0, UNIT * 0.32),
                    (-UNIT * 0.32, 0), (UNIT * 0.32, 0)]
            for r in range(HEIGHT):
                for c in range(WIDTH):
                    probs = self._arrows[r][c]
                    if not probs: continue
                    cx, cy = c * UNIT + UNIT // 2, r * UNIT + UNIT // 2
                    for i, p in enumerate(probs):
                        if p > 0:
                            self._arrow(cx, cy, cx + edge[i][0], cy + edge[i][1])

        for rect, btn in zip(self._btn_rects(), self.buttons):
            enabled = self._btn_enabled(btn)
            pressed = btn[0] == self._press_label and self._press_frames > 0
            if not enabled:
                bg, fg, border = (245, 245, 245), (170, 170, 170), (200, 200, 200)
            elif pressed:
                bg, fg, border = (160, 180, 220), BLACK, BLACK  # blue tint while held
            else:
                bg, fg, border = (220, 220, 220), BLACK, BLACK
            pygame.draw.rect(s, bg, rect)
            pygame.draw.rect(s, border, rect, 1)
            t = self._font.render(btn[0], True, fg)
            # Offset text slightly when pressed for a "depressed" feel.
            center = (rect.centerx + (1 if pressed else 0), rect.centery + (1 if pressed else 0))
            s.blit(t, t.get_rect(center=center))
        if self._press_frames > 0:
            self._press_frames -= 1

        pygame.display.flip()

    def _arrow(self, x0, y0, x1, y1):
        # Line from cell center to edge, then two short segments forming
        # the arrowhead — each ~30 degrees off the backward direction.
        pygame.draw.line(self._screen, BLACK, (x0, y0), (x1, y1), 2)
        ang = math.atan2(y1 - y0, x1 - x0)
        for sign in (-1, 1):
            a = ang + sign * 0.5
            pygame.draw.line(self._screen, BLACK, (x1, y1),
                             (x1 - 8 * math.cos(a), y1 - 8 * math.sin(a)), 2)
