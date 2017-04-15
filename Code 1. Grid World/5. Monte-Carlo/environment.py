import numpy as np

np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageTk, Image

UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('monte carlo')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.buildGraphic()
        self.texts = []

    def buildGraphic(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=HEIGHT * UNIT,
                                width=WIDTH * UNIT)

        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # image_load
        self.cat_image = ImageTk.PhotoImage(Image.open("../resources/rectangle.png").resize((65, 65), Image.ANTIALIAS))
        self.fire_image = ImageTk.PhotoImage(Image.open("../resources/triangle.png").resize((65, 65)))
        self.fish_image = ImageTk.PhotoImage(Image.open("../resources/circle.png").resize((65, 65)))

        # add image to canvas
        self.cat = self.canvas.create_image(50, 50, image=self.cat_image)
        self.hell1 = self.canvas.create_image(250, 150, image=self.fire_image)
        self.hell2 = self.canvas.create_image(150, 250, image=self.fire_image)
        self.fish = self.canvas.create_image(250, 250, image=self.fish_image)

        # pack all
        self.canvas.pack()

    def text_value(self, row, col, contents, action, font='Helvetica', size=10, style='normal', anchor="nw"):

        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        return self.texts.append(self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor))

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(0, 4):
                    state = [i, j]
                    if str(state) in q_table.index:
                        temp = q_table.ix[str(state), action]
                        self.text_value(j, i, round(temp, 2), action)

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.cat)
        origin = np.array([UNIT / 2, UNIT / 2])
        self.cat = self.canvas.create_image(50, 50, image=self.cat_image)
        # return observation
        return self.coords_to_state(self.canvas.coords(self.cat))

    def step(self, action):
        s = self.canvas.coords(self.cat)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.cat, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.cat)  # next state

        # reward function
        if s_ == self.canvas.coords(self.fish):
            reward = 100
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False

        s_ = self.coords_to_state(s_)

        return s_, reward, done

    def render(self):
        time.sleep(0.05)
        self.update()
