import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageTk,Image


UNIT = 100   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class GraphicDisplay(tk.Tk):
    def __init__(self):
        super(GraphicDisplay, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('qlearning')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.buildGraphic()

    def buildGraphic(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT): # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT): # 0~400 by 80
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)



        #image_load
        self.cat_image = ImageTk.PhotoImage(Image.open("../resources/cat.png").resize((50, 50), Image.ANTIALIAS))
        self.fire_image = ImageTk.PhotoImage(Image.open("../resources/fire.png").resize((50, 50)))
        self.fish_image = ImageTk.PhotoImage(Image.open("../resources/fish.png").resize((50, 50)))


        #add image to canvas
        self.cat =self.canvas.create_image(50, 50, image=self.cat_image)
        self.hell1 = self.canvas.create_image(250, 150, image=self.fire_image)
        self.hell2 = self.canvas.create_image(150, 250, image=self.fire_image)
        self.fish = self.canvas.create_image(250, 250, image=self.fish_image)

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.cat)
        origin = np.array([UNIT / 2, UNIT / 2])
        self.cat = self.canvas.create_image(50, 50, image=self.cat_image)
        # return observation
        return self.canvas.coords(self.cat)

    def step(self, action):
        s = self.canvas.coords(self.cat)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.cat, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.cat)  # next state

        # reward function
        if s_ == self.canvas.coords(self.fish):
            reward = 1
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.05)
        self.update()
