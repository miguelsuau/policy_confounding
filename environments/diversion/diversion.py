from hashlib import new
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class Diversion(gym.Env):

    NAME = 'diversion'
    
    CORRIDOR_LENGTH = 7

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    SLIPPERY = True

    OBS_SIZE = CORRIDOR_LENGTH + 1

    def __init__(self, seed, eval=False, random_action_prob=0.0, dynamic_programming=False):
        self.seed(seed)
        self.max_steps = 50
        self.img = None
        self.diversion = eval
        self.random_action_prob = random_action_prob
        self.dynamic_programming = dynamic_programming

    def reset(self):
        self.location = [0, 0]
        obs = np.zeros(self.CORRIDOR_LENGTH + 1, dtype=np.float32)
        obs[self.location[1]] = 1
        obs[-1] = self.location[0]
        self.steps = 0
        self.already_diverted = False
        return obs
    
    def step(self, action):
        
        if np.random.uniform(0,1) < self.random_action_prob:
            action = np.random.choice(len(self.ACTIONS))
        if not self.dynamic_programming:
            self.steps += 1
        if self.location[0] == 6:
            print("location 6")
        self.location = self.move(action)
        obs = np.zeros(self.CORRIDOR_LENGTH + 1, dtype=np.float32)
        obs[self.location[1]] = 1
        obs[-1] = self.location[0]

        reward, done = self.reward_done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        
        bitmap = self.bitmap
        bitmap[self.CORRIDOR_WIDTH:self.ROOM_WIDTH-self.CORRIDOR_WIDTH, self.CORRIDOR_WIDTH:] += 2
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
        
        if self.diversion and not self.already_diverted:
                if self.location[1] == self.CORRIDOR_LENGTH//2:
                    if self.location[0] == 0:
                        action = 1
                    else:
                        action = 0
                    self.already_diverted = True
                    
        if action == 0:
            new_location = [self.location[0]-1, self.location[1]]
        if action == 1:
            new_location = [self.location[0]+1, self.location[1]]
        if action == 2:
            new_location = [self.location[0], self.location[1]-1]
        if action == 3:
            new_location = [self.location[0], self.location[1]+1]
        
        
            
        if 0 <= new_location[0] < 2 and 0 <= new_location[1] < self.CORRIDOR_LENGTH:
            self.location = new_location
        else:
            new_location = self.location
        
        return new_location

        

    def reward_done(self):
        reward = -0.01
        done = False
        if self.location[1] == self.CORRIDOR_LENGTH - 1:
            if self.location[0] == 0:
                reward = 1.0
            if self.location[0] == 1:
                reward = 0.0
            done = True
            

        if self.steps >= self.max_steps:
            done = True

        return reward, done

            
                

