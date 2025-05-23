from hashlib import new
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class KeyDoor(gym.Env):

    NAME = 'keydoor'
    
    CORRIDOR_LENGTH = 7
    DOOR_LOCATION = 6
    KEY_LOCATION = 0
    START_LOCATION_TRAIN = 1
    START_LOCATION_EVAL = 5

    ACTIONS = {0: 'LEFT',
               1: 'RIGHT'}


    OBS_SIZE = CORRIDOR_LENGTH + 1

    POSITIVE_REWARD = 1.0
    NEGATIVE_REWARD = - 0.0

    def __init__(self, seed, eval=False, random_action_prob=0.0, dynamic_programming=False):
        # self.seed(seed)
        self.max_steps = 50
        self.img = None
        self.eval = eval
        self.random_action_prob = random_action_prob
        self.n_positive_rewards = 0
        self.n_negative_rewards = 0
        self.dynamic_programming = dynamic_programming

    def reset(self):
        if self.eval:
            self.location = self.START_LOCATION_EVAL
        else:
            self.location = self.START_LOCATION_TRAIN
        obs = np.zeros(self.CORRIDOR_LENGTH)
        obs[self.location] = 1
        self.steps = 0
        self.has_key = False
        obs = np.append(obs, int(self.has_key))
        self.start_location_eval_not_reached = True
        return obs
    
    def step(self, action):
        # if self.eval:
        #     if self.location != self.START_LOCATION_EVAL and self.start_location_eval_not_reached:
        #         action = 1
        self.location = self.move(action)
        # if self.eval:
        #     if self.location == self.START_LOCATION_EVAL:
        #         self.start_location_eval_not_reached = False
        if not self.dynamic_programming:
            self.steps += 1
        obs = np.zeros(self.CORRIDOR_LENGTH)
        obs[self.location] = 1
        if self.location == self.KEY_LOCATION:
            self.has_key = True
        obs = np.append(obs, int(self.has_key))
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
        # if seed is not None:
        #     np.random.seed(seed)
        pass

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
            new_location = self.location-1
        if action == 1:
            new_location = self.location+1

        if 0 <= new_location < self.CORRIDOR_LENGTH:
            self.location = new_location
        else:
            new_location = self.location

        return self.location

    def reward_done(self):
        reward = -.01
        done = False

        if self.location == self.DOOR_LOCATION:
            if self.has_key:
                reward = self.POSITIVE_REWARD
                done = True
                self.n_positive_rewards += 1
            else:
                reward = self.NEGATIVE_REWARD
                done = True
                self.n_negative_rewards += 1

        if self.steps >= self.max_steps:
            done = True

        return reward, done
    
    def get_test_obs(self):
        obs1 = np.zeros(self.CORRIDOR_LENGTH)
        obs1[self.START_LOCATION_EVAL] = 1
        obs1 = np.append(obs1, 1)

        obs2 = np.zeros(self.CORRIDOR_LENGTH)
        obs2[self.START_LOCATION_EVAL] = 1
        obs2 = np.append(obs2, 0)
        
        wrong_action = 1
        
        return obs1, obs2, wrong_action
                

