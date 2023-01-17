from hashlib import new
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class Tmaze(gym.Env):
    
    CORRIDOR_WIDTH = 2
    CORRIDOR_LENGTH = 7

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    SLIP_PROB = 1.0

    OBS_SIZE = CORRIDOR_LENGTH*CORRIDOR_WIDTH + 1

    def __init__(self, seed, eval=False):
        self.seed(seed)
        self.max_steps = 100
        self.img = None
        self.slippery = eval

    def reset(self):
        self.value = np.random.choice([-1,1],1)
        self.bitmap = np.zeros((self.CORRIDOR_WIDTH, self.CORRIDOR_LENGTH))
        # self.location = [np.random.choice(self.CORRIDOR_WIDTH), 0]#np.random.choice(self.CORRIDOR_LENGTH//2 - 1)]
        self.location = [0, 0]
        self.bitmap[self.location[0], self.location[1]] = 1
        obs = np.append(self.bitmap.flatten(), self.value)
        self.steps = 0
        return obs
    
    def step(self, action):
        self.steps += 1
        self.location, self.bitmap = self.move(action) 
        
        # if self.location[1] == 0:
        #     obs = np.append(self.bitmap.flatten(), self.value)
        # else:
        obs = np.append(self.bitmap.flatten(), 0)

        reward, done = self.reward_done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        
        bitmap = self.bitmap
        # bitmap[self.CORRIDOR_WIDTH:self.CORRIDOR_WIDTH-self.CORRIDOR_WIDTH, self.CORRIDOR_WIDTH:] += 2
        if self.img is None:
            fig, ax = plt.subplots(1)
            self.img = ax.imshow(self.bitmap)
        else:
            self.img.set_data(self.bitmap)
        # plt.pause(delay)
        plt.savefig('images/image.jpg')
        img = plt.imread('images/image.jpg')
        return img
        
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(self.OBS_SIZE,))

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        return spaces.Discrete(len(self.ACTIONS))

    def move(self, action):
        
        if action == 0:
            new_location = [self.location[0]+1, self.location[1]]
        if action == 1:
            new_location = [self.location[0]-1, self.location[1]]
        if action == 2:
            new_location = [self.location[0], self.location[1]-1]
        if action == 3:
            new_location = [self.location[0], self.location[1]+1]
        
        if self.slippery:
            # if new_location[1] == self.CORRIDOR_LENGTH - 2:
            if new_location[1] == self.CORRIDOR_LENGTH//2 and action == 3:
                if new_location[0] == 0:
                    new_location[0] = 1
                else:
                    new_location[0] = 0
            # if new_location[1] == self.CORRIDOR_LENGTH - 2 and self.location[1] == self.CORRIDOR_LENGTH - 3:# and (new_location[0] == 0 or new_location[0] == self.CORRIDOR_WIDTH -1):
            #     if new_location[0] == 0:
            #         new_location[0] = self.CORRIDOR_WIDTH-1
            #     else:
            #         new_location[0] = 0
                # new_location = [new_location[0], new_location[1] + shift]
            # if self.location[1] == self.CORRIDOR_LENGTH - 2:
            #     if self.location[0] != self.CORRIDOR_WIDTH//2:
            #         new_location = self.location
        # if  self.CORRIDOR_WIDTH <= new_location[1] <= self.CORRIDOR_LENGTH - 1 and \
        #     self.CORRIDOR_WIDTH <= new_location[0] <= self.CORRIDOR_WIDTH - self.CORRIDOR_WIDTH - 1:
            
        #     new_location =  self.location
            

        bitmap = np.zeros((self.CORRIDOR_WIDTH, self.CORRIDOR_LENGTH))
        if 0 <= new_location[0] < self.CORRIDOR_WIDTH and 0 <= new_location[1] < self.CORRIDOR_LENGTH:
            bitmap[new_location[0], new_location[1]] = 1
        else:
            bitmap = self.bitmap
            new_location = self.location
        
        return new_location, bitmap

    def reward_done(self):
        reward = -0.1
        done = False
        if self.location[1] == self.CORRIDOR_LENGTH - 1:
            if self.location[0] == 0:
                done = True
                if self.value == -1:
                    reward = 10.0
                # else:
                #     reward = -1.0
            if self.location[0] == self.CORRIDOR_WIDTH - 1:
                done = True
                if self.value == 1:
                    reward = 10.0
                # else:
                #     reward = -1.0

        if self.steps >= self.max_steps:
            done = True

        return reward, done

            
                

