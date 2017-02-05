import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageTk,Image
from agent import PolicyIterationAgent



UNIT = 100   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width

class GraphicDisplay(tk.Tk):
    def __init__(self):
        super(GraphicEnv, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('PolicyIteration')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT+50))
        self.texts = []
        self.arrows = []
        self.agent = PolicyIterationAgent()
        self._build_env()

    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        #Buttons
        iterationButton = tk.Button(self, text="Iterate", command=self.doIteration)
        iterationButton.configure(width=10, activebackground="#33B5E5")
        button1_window = self.canvas.create_window(MAZE_W*UNIT*0.13,(MAZE_H*UNIT)+10,window=iterationButton)

        policyButton = tk.Button(self, text="Policy Update", command=self.updatePolicy)
        policyButton.configure(width=10, activebackground="#33B5E5")
        button2_window = self.canvas.create_window(MAZE_W*UNIT*0.37,(MAZE_H*UNIT)+10,window=policyButton)

        policyButton = tk.Button(self, text="move", command=self.moveByPolicy)
        policyButton.configure(width=10, activebackground="#33B5E5")
        button3_window = self.canvas.create_window(MAZE_W * UNIT*0.62, (MAZE_H * UNIT) + 10, window=policyButton)

        policyButton = tk.Button(self, text="clear", command=self.clear)
        policyButton.configure(width=10, activebackground="#33B5E5")
        button4_window = self.canvas.create_window(MAZE_W * UNIT * 0.87, (MAZE_H * UNIT) + 10, window=policyButton)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT): # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT): # 0~400 by 80
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        #image_load
        self.up_image = ImageTk.PhotoImage(Image.open("../resources/up.png").resize((13,13)))
        self.right_image = ImageTk.PhotoImage(Image.open("../resources/right.png").resize((13, 13)))
        self.left_image = ImageTk.PhotoImage(Image.open("../resources/left.png").resize((13, 13)))
        self.down_image = ImageTk.PhotoImage(Image.open("../resources/down.png").resize((13, 13)))
        self.cat_image = ImageTk.PhotoImage(Image.open("../resources/cat.png").resize((50, 50), Image.ANTIALIAS))
        self.fire_image = ImageTk.PhotoImage(Image.open("../resources/fire.png").resize((50, 50)))
        self.fish_image = ImageTk.PhotoImage(Image.open("../resources/fish.png").resize((50, 50)))


        #add image to canvas
        self.cat =self.canvas.create_image(50, 50, image=self.cat_image)
        self.hell1 = self.canvas.create_image(250, 150, image=self.fire_image)
        self.hell2 = self.canvas.create_image(150, 250, image=self.fire_image)
        self.fish = self.canvas.create_image(250, 250, image=self.fish_image)

        # add reward text
        self.text_reward(2, 2, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")

        # pack all
        self.canvas.pack()

    def clear(self):
        for i in self.texts:
            self.canvas.delete(i)

        for i in self.arrows:
            self.canvas.delete(i)

        self.canvas.delete(self.cat)
        self.cat = self.canvas.create_image(50, 50, image=self.cat_image)
        self.agent = PolicyIterationAgent()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.cat)
        origin = np.array([UNIT/2, UNIT/2])
        self.cat = self.canvas.create_image(50, 50, image=self.cat_image)
        # return observation
        return self.canvas.coords(self.cat)

    def text_value(self, row , col , contents, font='Helvetica', size=12, style='normal', anchor="nw"):
        origin_x, origin_y = 85, 70
        x , y = origin_y+(UNIT*(col)),origin_x+(UNIT*(row))
        font = (font, str(size), style)
        return self.texts.append(self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor))

    def text_reward(self, row, col, contents, font='Helvetica', size=12, style='normal', anchor="nw"):
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT * (col)),origin_x + (UNIT * (row))
        font = (font, str(size), style)
        return self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor)

    def step(self, action):
        s = self.canvas.coords(self.cat)

        base_action = np.array([0, 0])
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
        if s_== self.canvas.coords(self.fish):
            reward = 1
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    def catMove(self,action):
        s = self.canvas.coords(self.cat)
        base_action = np.array([0, 0])
        self.render()
        if action[0]==1: #down
            base_action[1] += UNIT
        elif action[0] == -1: #up
            base_action[1] -= UNIT
        elif action[1] == 1: #right
            base_action[0] += UNIT
        elif action[1] == -1: #left
            base_action[0] -= UNIT

        self.canvas.move(self.cat, base_action[0], base_action[1])  # move agent

    def catLocation(self):
        temp = self.canvas.coords(self.cat)
        x = (temp[0]/100)-0.5
        y = (temp[1]/100) - 0.5
        return (int(y),int(x))

    def moveByPolicy(self):
        self.canvas.delete(self.cat)
        self.cat = self.canvas.create_image(50, 50, image=self.cat_image)
        while len(self.agent.getPolicies()[self.catLocation()[0]][self.catLocation()[1]])!=0:
            self.after(100,self.catMove(self.agent.getPolicies()[self.catLocation()[0]][self.catLocation()[1]][0]))

    def drawOneArrow(self,col,row,action):
        if action[0]==1: #down
            origin_x, origin_y = 50+(UNIT*row), 90+(UNIT*col)
            self.arrows.append(self.canvas.create_image(origin_x,origin_y, image=self.down_image))

        elif action[0] == -1: #up
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.up_image))

        elif action[1] == 1: #right
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.right_image))

        elif action[1] == -1: #left
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.left_image))
        else :
            print("Not proper action ")

    def drawFromPolicy(self,policies):

        for i in range(MAZE_H):
            for j in range(MAZE_W):
                for k in policies[i][j]:
                    self.drawOneArrow(i,j,k)

    def printValues(self, values):
        for i in range(MAZE_W):
            for j in range(MAZE_H):
                self.text_value(i, j, values[i][j])

    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.cat)
        self.update()

    def doIteration(self):
        for i in self.texts:
            self.canvas.delete(i)
        self.agent.doIteration(1)
        self.printValues(self.agent.getValues())

    def updatePolicy(self):
        for i in self.arrows:
            self.canvas.delete(i)
        self.agent.updatePolicy()
        self.drawFromPolicy(self.agent.getPolicies())



if __name__ == "__main__":
    gridworld = GraphicDisplay()
    gridworld.mainloop()
