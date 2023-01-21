from hashlib import new
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class Tmaze(gym.Env):

    NAME = 'tmaze'
    
    CORRIDOR_WIDTH = 3
    CORRIDOR_LENGTH = 7

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    RANDOM_ACTION_PROB = 0.2
    BITMAP = True
    
    if BITMAP:
        OBS_SIZE = CORRIDOR_LENGTH*CORRIDOR_WIDTH + 1
    else:
        OBS_SIZE = 3

    def __init__(self, seed, eval=False, stochasticity=False):
    
        # self.seed(seed)
        self.max_steps = 50
        self.img = None
        self.slippery = eval
        self.stochasticity = stochasticity
        self.fixed_value = False
        self.signal = None
        self.fixed_start = False

    def reset(self):
        if not self.fixed_value:
            self.signal = np.random.choice([-1,1])
        else:
            print(
                '''
                WARNING: The initial signal is fixed. This is only meant for evaluating
                the policy. Set the variable fixed_value = False for training.
                '''    
            )
        # if not self.fixed_start:
        # self.location = [np.random.choice([0, self.CORRIDOR_WIDTH-1]), 0]#np.random.choice(self.CORRIDOR_LENGTH//2 - 1)]
        self.start_location = [np.random.choice(self.CORRIDOR_WIDTH), 0]#np.random.choice(self.CORRIDOR_LENGTH//2 - 1)]
        self.start_location = [1, 0]
        self.location = self.start_location
        # else:
        #     self.location = [0, 0]
        # self.location = [1, 0]
        # obs = np.append(self.bitmap.flatten(), self.signal)
        obs = self.get_obs(signal=self.signal)
        if self.slippery:
            self.ice_location = np.random.choice(np.arange(2, self.CORRIDOR_LENGTH -1))

        self.steps = 0
        self.override = 0
        return obs
    
    def step(self, action):
        self.steps += 1
        reward, done = self.reward_done(action)
        
        # if self.slippery:
            # print(self.location, action, self.signal, reward, done)
        # if not done:
        self.move(action)
        if self.location[0] == self.start_location[0] and self.location[1] == self.start_location[1]:# and self.location[0] == 0:
            # obs = np.append(self.bitmap.flatten(), self.signal)
            obs = self.get_obs(signal=self.signal)
        else:
            # obs = np.append(self.bitmap.flatten(), 0)
        
            obs = self.get_obs(signal=0)

        # else:
        #     # obs = np.append(np.zeros((self.CORRIDOR_WIDTH, self.CORRIDOR_LENGTH)), 0)
        #     obs = self.get_obs(signal=0)

        return obs, reward, done, {}

    def render(self, mode='human'):
        
        import time
        # bitmap[self.CORRIDOR_WIDTH:self.CORRIDOR_WIDTH-self.CORRIDOR_WIDTH, self.CORRIDOR_WIDTH:] += 2
        if self.img is None:
            fig, ax = plt.subplots(1)
            self.img = ax.imshow(self.bitmap)
        else:
            self.img.set_data(self.bitmap)
        # plt.pause(delay)
        # plt.show()
        plt.savefig('images/image.jpg')
        img = plt.imread('images/image.jpg')
        time.sleep(.5)
        return img
        
    def seed(self, seed=None):
        # if seed is not None:
        #     np.random.seed(seed)
        pass

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=(self.OBS_SIZE,))

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        return spaces.Discrete(len(self.ACTIONS))

    def get_obs(self, signal):
        if self.BITMAP:
            self.bitmap = np.zeros((self.CORRIDOR_WIDTH, self.CORRIDOR_LENGTH))
            self.bitmap[self.location[0], self.location[1]] = 1
            return np.append(self.bitmap.flatten(), signal)
        else:
            return np.array((
                    self.location[0]/(self.CORRIDOR_WIDTH-1), 
                    self.location[1]/(self.CORRIDOR_LENGTH-1), 
                    signal
                    ))
    
    def set_fixed_value(self, value=None):
        self.fixed_value = True
        self.signal = value

    def set_fixed_start(self):
        self.fixed_start = True

    def set_random_value(self):
        self.fixed_value = False

    def set_random_start(self):
        self.fixed_start = False

    def move(self, action):

        if self.stochasticity:
            if np.random.uniform(0,1) < self.RANDOM_ACTION_PROB:
                action = np.random.choice(range(0,len(self.ACTIONS)))

        if self.slippery: #Override action
            # if new_location[1] == self.CORRIDOR_LENGTH - 2:
            # self.ice_location = self.CORRIDOR_LENGTH//2
            if self.location[1] == self.ice_location and self.override < self.CORRIDOR_WIDTH - 1:
                # print('ice location: ', self.ice_location)
                self.override += 1
                # print('slipped')
                if self.location[0] == 0:
                    # new_location[0] = self.CORRIDOR_WIDTH - 1
                    action = 1
                    # new_location[0] = 1
                elif self.location[0] == self.CORRIDOR_WIDTH - 1:
                    action = 0
                    # new_location[0] = 1
                else:
                    self.override += 1
                    if self.signal == -1:
                        # new_location[0] = self.CORRIDOR_WIDTH - 1
                        action = 1
                    else:
                        # new_location[0] = 0
                        action = 0
            
        
        if action == 0:
            new_location = [self.location[0]-1, self.location[1]]
        if action == 1:
            new_location = [self.location[0]+1, self.location[1]]
        if action == 2:
            new_location = [self.location[0], self.location[1]-1]
        if action == 3:
            new_location = [self.location[0], self.location[1]+1]
        
        # bitmap = np.zeros((self.CORRIDOR_WIDTH, self.CORRIDOR_LENGTH))
        if 0 <= new_location[0] < self.CORRIDOR_WIDTH and 0 <= new_location[1] < self.CORRIDOR_LENGTH:
            self.location = new_location

        # if 1 <= new_location[0] < self.CORRIDOR_WIDTH - 1 and 0 <= new_location[1] < self.CORRIDOR_LENGTH:
        #     self.location = new_location
        
        # if new_location[1] == self.CORRIDOR_LENGTH - 1:  
        #     if new_location[0] == 0 or new_location[0] == self.CORRIDOR_WIDTH - 1:
        #         self.location = new_location
        # else:
        #     # bitmap = self.bitmap
        #     new_location = self.location
        
        # return new_location

    def reward_done(self, action):
        # reward = -0.01
        reward = 0.0
        done = False
        if self.location[1] == self.CORRIDOR_LENGTH - 1:
            if self.location[0] == 0 and action == 0:
                done = True
                if self.signal == -1:
                    reward = 1.0
                else:
                    reward = -1.0
            if self.location[0] == self.CORRIDOR_WIDTH-1 and action == 1:
                done = True
                if self.signal == 1:
                    reward = 1.0
                else:
                    reward = -1.0
            # if self.slippery:
            #     print(self.location, action, self.signal, reward, done)
        if self.steps >= self.max_steps:
            done = True

        return reward, done

    def optimal_policy(self, value):
        if value == -1:
            return [3]*(self.CORRIDOR_LENGTH-1) + [0]
        else:
            return [1] + [3]*(self.CORRIDOR_LENGTH-1) + [1]




    
                

