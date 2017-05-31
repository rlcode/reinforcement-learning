import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from PIL.ImageTK import PhotoImage

np.random.seed(1)

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
        self.rect_img = PhotoImage(
            Image.open("../img/rect.png").resize((65, 65), Image.ANTIALIAS))
        self.trig_img = PhotoImage(
            Image.open("../img/trig.png").resize((65, 65)))
        self.circ_img = PhotoImage(
            Image.open("../img/circ.png").resize((65, 65)))

        # add img to canvas
        self.rect = self.canvas.create_image(50, 50, image=self.rect_img)
        self.trig1 = self.canvas.create_image(250, 150, image=self.trig_img)
        self.trig2 = self.canvas.create_image(150, 250, image=self.trig_img)
        self.circ = self.canvas.create_image(250, 250, image=self.circ_img)

        # pack all
        self.canvas.pack()

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
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
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

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
        self.canvas.delete(self.rect)
        origin = np.array([UNIT / 2, UNIT / 2])
        self.rect = self.canvas.create_image(50, 50, image=self.rect_img)
        # return observation
        return self.coords_to_state(self.canvas.coords(self.rect))

    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1]) # move agent

        next_state = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_state == self.canvas.coords(self.circ):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.trig1),
                            self.canvas.coords(self.trig2)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)

        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()
