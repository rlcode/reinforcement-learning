import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from PIL.ImageTK import PhotoImage

UNIT = 50  # pixels
HEIGHT = 10  # grid height
WIDTH = 10  # grid width

# np.random.seed(1)


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('Policy Gradient')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.build_graphic()
        self.counter = 0

    def build_graphic(self):
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
        self.rectangle_img = PhotoImage(
            Image.open("../img/rectangle.png").resize((30, 30), Image.ANTIALIAS))
        self.triangle_img = PhotoImage(
            Image.open("../img/triangle.png").resize((30, 30)))
        self.circle_img = PhotoImage(
            Image.open("../img/circle.png").resize((30, 30)))

        self.rewards = list()
        self.goal = list()

        # obstacle
        self.set_reward([2, 7], -1)
        self.set_reward([3, 2], -1)
        self.set_reward([2, 5], -1)
        self.set_reward([4, 9], -1)
        self.set_reward([5, 7], -1)
        self.set_reward([6, 4], -1)
        self.set_reward([7, 8], -1)
        self.set_reward([8, 3], -1)
        self.set_reward([9, 1], -1)

        # #goal
        self.set_reward([9, 9], 5)

        # add img to canvas
        self.rectangle = self.canvas.create_image(UNIT/2, UNIT/2, image=self.rectangle_img)

        # pack all`
        self.canvas.pack()

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()
        # obstacle
        self.set_reward([2, 7], -1)
        self.set_reward([3, 2], -1)
        self.set_reward([2, 5], -1)
        self.set_reward([4, 9], -1)
        self.set_reward([5, 7], -1)
        self.set_reward([6, 4], -1)
        self.set_reward([7, 8], -1)
        self.set_reward([8, 3], -1)
        self.set_reward([9, 1], -1)

        # #goal
        self.set_reward([9, 9], 5)

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        temp = {}
        if reward > 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(
                UNIT * state[0] + UNIT/2,
                UNIT * state[1] + UNIT/2,
                image=self.circle_img)
            self.goal.append(temp['figure'])


        elif reward < 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(
                UNIT * state[0] + UNIT/2,
                UNIT * state[1] + UNIT/2,
                image=self.triangle_img)

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

    # new methods

    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0
        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']
                if reward['reward'] == 5:
                    check_list['if_goal'] = True
        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_image(UNIT/2, UNIT/2, image=self.rectangle_img)
        # return observation

        self.reset_reward()
        return self.get_state()

    def step(self, action):
        self.counter += 1
        self.render()

        next_coords = self.move(self.rectangle, action)

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_goal']
        reward = check['rewards']

        s_ = self.get_state()

        return s_, reward, done

    def get_state(self):
        agent_location = self.coords_to_state(self.canvas.coords(self.rectangle))
        agent_x = agent_location[0]
        agent_y = agent_location[1]

        locations = list()

        locations.append(agent_x)
        locations.append(agent_y)

        for reward in self.rewards:
            reward_location = reward['state']
            locations.append(agent_x - reward_location[0])
            locations.append(agent_y - reward_location[1])

        return locations

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 10:
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp['figure'])
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)
        return new_rewards

    def move_const(self, target):
        s = self.canvas.coords(target)
        base_action = np.array([0, 0])

        if s[0] < (WIDTH - 1) * UNIT:
            base_action[0] += UNIT
        else:
            base_action[0] = -(WIDTH - 1) * UNIT

        # if action == 4 # move _none

        if target is not self.rectangle and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]:
            base_action = np.array([0, 0])

        self.canvas.move(target, base_action[0], base_action[1])
        s_ = self.canvas.coords(target)
        return s_

    def move(self, target, action):
        s = self.canvas.coords(target)
        base_action = np.array([0, 0])

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        if target is not self.rectangle and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]:
            base_action = np.array([0, 0])

        self.canvas.move(target, base_action[0], base_action[1])
        s_ = self.canvas.coords(target)
        return s_

    def render(self):
        time.sleep(0.1)
        self.update()
