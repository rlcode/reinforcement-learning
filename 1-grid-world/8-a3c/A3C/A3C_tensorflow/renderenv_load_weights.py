import numpy as np
import Tkinter as tk
from PIL import ImageTk, Image
import time

PhotoImage = ImageTk.PhotoImage

UNIT = 50  # pixels

HEIGHT = 0
WIDTH = 0

# Goal position
goal_x = 0 # x position of goal
goal_y = 0 # y position of goal

np.random.seed(1)

class EnvRender(tk.Tk):
    def __init__(self, gx, gy, Hx, Hy):
        tk.Tk.__init__(self)
        global HEIGHT
        global WIDTH

        HEIGHT = Hx
        WIDTH = Hy

        self.title('A3C')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()

        self.counter = 0
        self.objects = []

        # obstacle
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)
        # #goal
        global goal_x
        global goal_y
        goal_x = gx
        goal_y = gy

        self.set_reward([goal_x, goal_y], 1)

    def _build_canvas(self):

        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.objects = []
        self.goal = []
        # add image to canvas
        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0])

        # pack all`
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((30, 30)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((30, 30)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((30, 30)))

        return rectangle, triangle, circle

    def reset_object(self):
        for obj in self.objects:
            self.canvas.delete(obj['id'])

        self.objects = []
        # obstacle
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)
        # #goal

        self.set_reward([goal_x, goal_y], 1)

    def set_reward(self, state, reward):
    	state = [int(state[0]), int(state[1])]
    	x = int(state[0])
    	y = int(state[1])

    	tmp = {}
    	if reward > 0:
    		tmp['id'] = self.canvas.create_image((UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2, image=self.shapes[2])
    	elif reward < 0:
    		tmp['id'] = self.canvas.create_image((UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2, image=self.shapes[1])

    	tmp['reward'] = reward
    	self.objects.append(tmp)

    def reset(self, oenv):
    	self.update()
    	time.sleep(0.5)

    	self.canvas.delete(self.rectangle)
    	self.rectangle = self.canvas.create_image(oenv.rectangle[0], oenv.rectangle[1], image=self.shapes[0])
    	self.reset_object()

    def move(self, next_coords, mod_rewards):
   		self.render()
   		self.counter += 1

   		if self.counter % 2 == 1:
   			for obj in self.objects:
   				if obj['reward'] < 0:
   					self.canvas.delete(obj['id'])
   			
   			self.objects = [item for item in self.objects if item['reward'] > 0]

	   		for item in mod_rewards:
	   			if item['reward'] < 0:
	   				tmp = {}
	   				tmp['id'] = self.canvas.create_image(item['figure'][0], item['figure'][1], image=self.shapes[1])
	   				tmp['reward'] = item['reward']
	   				self.objects.append(tmp)

   		self.canvas.delete(self.rectangle)
   		self.rectangle = self.canvas.create_image(next_coords[0], next_coords[1], image=self.shapes[0])
   		self.canvas.tag_raise(self.rectangle)


    def render(self):
    	time.sleep(0.07)
    	self.update()

