import time
import numpy as np
import random
from PIL import  Image
UNIT = 50  # pixels

HEIGHT = 0 # grid height
WIDTH = 0 # grid width

# Goal position
goal_x = 0 # x position of goal
goal_y = 0 # y position of goal

#obstacle y locations
poss = []
obs_list = []
y_list = []
# possible grid sizes
grid_size = [7, 9, 11, 13, 15]
#np.random.seed(1)

class Env(object):
    def __init__(self):
        super(Env, self).__init__()
        global HEIGHT
        global WIDTH

        HEIGHT = random.choice(grid_size)
        WIDTH = random.choice(grid_size)
        
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)

        self.counter = 0
        self.rewards = []
        self.goal = []
        
        # rectangle
        self.rectangle = (UNIT/2, UNIT/2)

        # obstacle
        global obs_list
        global poss 
        global y_list

        obs_list = random.sample(xrange(1, HEIGHT), 5)
        y_list = [i for i in range(HEIGHT)]

        obs_list.sort()

        self.set_reward([0, obs_list[0]], -1)
        self.set_reward([WIDTH-1, obs_list[1]], -1)
        self.set_reward([1, obs_list[2]], -1)
        self.set_reward([WIDTH-2, obs_list[3]], -1)
        self.set_reward([2, obs_list[4]], -1)
        # #goal
        global goal_x
        global goal_y

        poss = list(set(y_list) - set(obs_list))

        goal_x = random.randint(0, WIDTH-1)
        goal_y = random.choice(poss)

        self.set_reward([goal_x, goal_y], 1)

    def reset_reward(self):

        self.rewards = []
        self.goal = []

        HEIGHT = random.choice(grid_size)
        WIDTH = random.choice(grid_size)

        obs_list = random.sample(xrange(1, HEIGHT), 5)
        obs_list.sort()

        self.set_reward([0, obs_list[0]], -1)
        self.set_reward([WIDTH-1, obs_list[1]], -1)
        self.set_reward([1, obs_list[2]], -1)
        self.set_reward([WIDTH-2, obs_list[3]], -1)
        self.set_reward([2, obs_list[4]], -1)

        poss = list(set(y_list) - set(obs_list))

        goal_x = random.randint(0, WIDTH-1)
        goal_y = random.choice(poss)

        self.set_reward([goal_x, goal_y], 1)


    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}

        if reward > 0:
            temp['reward'] = reward

            temp['figure'] = ((UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2)
            self.goal.append(temp['figure'])


        elif reward < 0:
            temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = ((UNIT * x) + UNIT / 2,(UNIT * y) + UNIT / 2)

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
                if reward['reward'] == 1:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self):
        x, y = self.rectangle
        
        tmp_x = self.rectangle[0] + UNIT / 2 - x
        tmp_y = self.rectangle[1] + UNIT / 2 - y

        self.rectangle = (tmp_x, tmp_y)

        # return observation
        self.reset_reward()
        return self.get_state()

    def step(self, action):
        self.counter += 1

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action)
        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_goal']
        reward = check['rewards']

        s_ = self.get_state()

        return s_, reward, done, next_coords, self.rewards

    def get_state(self):

        location = self.coords_to_state(self.rectangle)
        agent_x = location[0]
        agent_y = location[1]

        states = list()

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y)
            if reward['reward'] < 0:
                states.append(-1)
                states.append(reward['direction'])
            else:
                states.append(1)

        return states

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 1:
                new_rewards.append(temp)
                continue
            temp['figure'] = self.move_const(temp)
            temp['state'] = self.coords_to_state(temp['figure'])
            new_rewards.append(temp)
        return new_rewards

    def move_const(self, target):
        s = target['figure']
        base_action = np.array([0, 0])

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:
            target['direction'] = 1
        elif s[0] == UNIT / 2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT

        if((target['figure'][0] != self.rectangle[0] or target['figure'][1] != self.rectangle[1]) 
           and  s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        tmp_x = target['figure'][0] + base_action[0]
        tmp_y = target['figure'][1] + base_action[1]
        target['figure'] = (tmp_x, tmp_y)

        s_ = target['figure']

        return s_

    def move(self, target, action):
        s = target
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
        
        tmp_x = target[0] + base_action[0]
        tmp_y = target[1] + base_action[1]

        target = (tmp_x, tmp_y)
        self.rectangle = (tmp_x, tmp_y)

        s_ = target

        return s_
