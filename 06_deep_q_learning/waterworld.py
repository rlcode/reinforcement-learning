import pygame
import sys
import math

#import .base
from ple.games.base.pygamewrapper import PyGameWrapper

from ple.games.utils.vec2d import vec2d
from ple.games.utils import percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d
from ple.games.primitives import Player, Creep


class WaterWorld(PyGameWrapper):
    """
    Based Karpthy's WaterWorld in `REINFORCEjs`_.
    .. _REINFORCEjs: https://github.com/karpathy/reinforcejs
    Parameters
    ----------
    width : int
        Screen width.
    height : int
        Screen height, recommended to be same dimension as width.
    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    """

    def __init__(self,
                 width=48,
                 height=48,
                 num_creeps=3):

        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_CREEPS = num_creeps
        self.CREEP_TYPES = ["GOOD", "BAD"]
        self.CREEP_COLORS = [(40, 140, 40), (150, 95, 95)]
        radius = percent_round_int(width, 0.047)
        self.CREEP_RADII = [radius, radius]
        self.CREEP_REWARD = [
            self.rewards["positive"],
            self.rewards["negative"]]
        self.CREEP_SPEED = 0.25 * width
        self.AGENT_COLOR = (60, 60, 140)
        self.AGENT_SPEED = 0.25 * width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = (self.width / 2, self.height / 2)

        self.creep_counts = {
            "GOOD": 0,
            "BAD": 0
        }

        self.dx = 0
        self.dy = 0
        self.player = None
        self.creeps = None

    def _handle_player_events(self):
        self.dx = 0
        self.dy = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx -= self.AGENT_SPEED

                if key == self.actions["right"]:
                    self.dx += self.AGENT_SPEED

                if key == self.actions["up"]:
                    self.dy -= self.AGENT_SPEED

                if key == self.actions["down"]:
                    self.dy += self.AGENT_SPEED

    def _add_creep(self, type=None):
        if type == None:
            creep_type = self.rng.choice([0, 1])
        else:
            creep_type = type

        creep = None
        pos = (0, 0)
        dist = 0.0

        while dist < 1.5:
            radius = self.CREEP_RADII[creep_type] * 1.5
            pos = self.rng.uniform(radius, self.height - radius, size=2)
            dist = math.sqrt(
                (self.player.pos.x - pos[0])**2 + (self.player.pos.y - pos[1])**2)

        creep = Creep(
            self.CREEP_COLORS[creep_type],
            self.CREEP_RADII[creep_type],
            pos,
            self.rng.choice([-1, 1], 2),
            0.2 * self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type],
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            self.rng.rand()
        )

        self.creeps.add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def getGameState(self):
        """
        Returns
        -------
        dict
            * player x position.
            * player y position.
            * player x velocity.
            * player y velocity.
            * player distance to each creep
        """
        #
        # state = {
        #     "player_x": self.player.pos.x,
        #     "player_y": self.player.pos.y,
        #     "player_velocity_x": self.player.vel.x,
        #     "player_velocity_y": self.player.vel.y,
        #     "creep_dist": {
        #         "GOOD": [],
        #         "BAD": []
        #     }
        # }

        state = {

            "creeps": [],

            "player_x": self.player.pos.x,
            "player_y": self.player.pos.y,
            "player_velocity_x": self.player.vel.x,
            "player_velocity_y": self.player.vel.y

        }

        for c in self.creeps:
            state["creeps"].append(self.player.pos.x - c.pos.x)
            state["creeps"].append(self.player.pos.y - c.pos.y)
            if c.TYPE == "GOOD" :
                state["creeps"].append(1)
            else:
                state["creeps"].append(-1)
        return state

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.creep_counts = {"GOOD": 0, "BAD": 0}

        if self.player is None:
            self.player = Player(
                self.AGENT_RADIUS, self.AGENT_COLOR,
                self.AGENT_SPEED, self.AGENT_INIT_POS,
                self.width, self.height
            )

        else:
            self.player.pos = vec2d(self.AGENT_INIT_POS)
            self.player.vel = vec2d((0.0, 0.0))

        if self.creeps is None:
            self.creeps = pygame.sprite.Group()
        else:
            self.creeps.empty()

        for i in range(int(self.N_CREEPS / 2)):
            self._add_creep(type = 1)

        for i in range(int(self.N_CREEPS / 2)):
            self._add_creep(0)

        self.score = 0
        self.ticks = 0
        self.lives = -1

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)

        hits = pygame.sprite.spritecollide(self.player, self.creeps, True)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            if creep.TYPE == "GOOD":
                type = 1
            else:
                type = 0

            self._add_creep(1 - type)

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.creeps.update(dt)

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = WaterWorld(width=256, height=256, num_creeps=10)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
