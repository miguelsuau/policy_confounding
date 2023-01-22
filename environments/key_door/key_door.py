from hashlib import new
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class KeyDoor(gym.Env):
    
    CORRIDOR_LENGTH = 7
    DOOR_LOCATION = 6
    KEY_LOCATION = 0
    START_LOCATION_TRAIN = 1
    START_LOCATION_EVAL = 5

    ACTIONS = {0: 'LEFT',
               1: 'RIGHT'}


    OBS_SIZE = CORRIDOR_LENGTH

    def __init__(self, seed, eval, random_action_prob=0.0):
        self.seed(seed)
        self.max_steps = 50
        self.img = None
        self.eval = eval
        self.random_action_prob = random_action_prob

    def reset(self):
        if self.eval:
            self.location = self.START_LOCATION_EVAL
        else:
            self.location = self.START_LOCATION_TRAIN
        obs = np.zeros(self.CORRIDOR_LENGTH)
        obs[self.location] = 1
        self.steps = 0
        self.has_key = False
        return obs
    
    def step(self, action):
        self.steps += 1
        self.location = self.move(action) 
        obs = np.zeros(self.CORRIDOR_LENGTH)
        obs[self.location] = 1
        if self.location == self.KEY_LOCATION:
            self.has_key = True
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

        if np.random.uniform(0,1) < self.random_action_prob:
            action = np.random.choice(len(self.ACTIONS))
        
        if action == 0:
            new_location = self.location+1
        if action == 1:
            new_location = self.location-1

        if new_location == self.DOOR_LOCATION:
            if self.has_key:
                self.location = new_location
            else:
                new_location = self.location     
        elif 0 <= new_location < self.CORRIDOR_LENGTH:
            self.location = new_location
        else:
            new_location = self.location
        
        return new_location

    def reward_done(self):
        reward = -.01
        done = False
        if self.location == self.CORRIDOR_LENGTH - 1:
            reward = 1.0
            done = True
        if self.steps >= self.max_steps:
            done = True

        return reward, done

            
                

