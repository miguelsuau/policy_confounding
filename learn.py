from logging import INFO
import numpy as np
import os
import sys
sys.path.append("..")

import gym
import sacred
from sacred.observers import MongoObserver
import pymongo
from sshtunnel import SSHTunnelForwarder
import csv
from copy import deepcopy
import time

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, PPO
from callback import Callback
from stable_baselines3.common.buffers import ReplayBuffer


def generate_path(path):
    """
    Generate a path to store e.g. logs, models and plots. Check if
    all needed subpaths exist, and if not, create them.
    """
    result_path = os.path.join("../results", path)
    model_path = os.path.join("../models", path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return path


def add_mongodb_observer():
    """
    connects the experiment instance to the mongodb database
    """
    MONGO_HOST = 'TUD-tm2'
    MONGO_DB = 'policy_confounding'
    PKEY = '~/.ssh/id_rsa'
    
    print("Trying to connect to mongoDB '{}'".format(MONGO_DB))
    server = SSHTunnelForwarder(
        MONGO_HOST,
        ssh_pkey=PKEY,
        remote_bind_address=('127.0.0.1', 27017)
        )
    server.start()
    DB_URI = 'mongodb://localhost:{}/policy_confounding'.format(server.local_bind_port)
        
    
    # DB_URI = 'mongodb://localhost:27017/policy_confounding'


    ex.observers.append(MongoObserver.create(DB_URI, db_name=MONGO_DB, ssl=False))
    print("Added MongoDB observer on {}.".format(MONGO_DB))
        # print("ONLY FILE STORAGE OBSERVER ADDED")
        # from sacred.observers import FileStorageObserver
        # ex.observers.append(FileStorageObserver.create('saved_runs'))

class Experiment(object):
    """
    Creates experiment object to interact with the environment and
    the agent and log results.
    """
    def __init__(self, parameters, _run, seed):
        self._run = _run
        self._seed = seed
        self.parameters = parameters['main']
        self.seed = seed
        self.log_dir = 'tmp/'
        os.makedirs(self.log_dir, exist_ok=True)
        self.create_env()

    def create_env(self):
        
        env_id = self.parameters['name']
        
        env = DummyVecEnv([lambda: gym.make(env_id, seed=np.random.randint(1.0e+6), eval=False)])
        if self.parameters['n_stack'] > 1:
            env = VecFrameStack(env, n_stack=self.parameters['n_stack'])
        self.original_env = VecMonitor(env, self.log_dir)

        env = DummyVecEnv([lambda: gym.make(env_id, seed=np.random.randint(1.0e+6), eval=False, stochasticity=self.parameters['stochasticity'])])
        if self.parameters['n_stack'] > 1:
            env = VecFrameStack(env, n_stack=self.parameters['n_stack'])
        self.train_env = VecMonitor(env, self.log_dir)
        

        env = DummyVecEnv([lambda: gym.make(env_id, seed=np.random.randint(1.0e+6), eval=True)])
        if self.parameters['n_stack'] > 1:
            env = VecFrameStack(env, n_stack=self.parameters['n_stack'])
        self.eval_env = VecMonitor(env, self.log_dir)
    
    def make_env(self, env_id, rank, seed=0, eval=False):
        """
        Utility function for multiprocessed env.
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id, seed=seed+np.random.randint(1.0e+6), eval=eval)
            # env = Monitor(env, './logs')
            env.seed(seed + rank)
            return env
        return _init   
           
    def linear_schedule(self, initial_value: float, final_value: float):
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * (initial_value - final_value) + final_value

        return func

    def learn(self):
        
        policy_kwargs = dict(net_arch=[
            self.parameters['hidden_size'], 
            self.parameters['hidden_size_2']
            ])
        if self.parameters['algorithm'] == 'PPO':
            self.agent = PPO(
                "MlpPolicy", 
                self.train_env, 
                verbose=0, 
                clip_range=self.parameters['epsilon'], 
                clip_range_vf=self.parameters['epsilon'],
                vf_coef=self.parameters['c1'],
                batch_size=self.parameters['batch_size'],
                n_steps=self.parameters['rollout_steps'],
                ent_coef=self.parameters['beta'], 
                n_epochs=self.parameters['num_epoch'],
                learning_rate=self.linear_schedule(self.parameters['learning_rate'], self.parameters['learning_rate_final']),
                # learning_rate=self.parameters['learning_rate'],
                gamma=self.parameters['gamma'],
                policy_kwargs=policy_kwargs
                )

        elif self.parameters['algorithm'] == 'DQN':
            self.agent = DQN(
                "MlpPolicy", 
                self.train_env, 
                verbose=0,
                buffer_size=int(self.parameters['buffer_size']), 
                learning_starts=self.parameters['learning_starts'],
                batch_size=self.parameters['batch_size'],
                learning_rate=self.linear_schedule(self.parameters['learning_rate'], self.parameters['learning_rate_final']), 
                # learning_rate=self.parameters['learning_rate'],
                target_update_interval=self.parameters['target_update_interval'],
                exploration_initial_eps=self.parameters['exploration_initial_eps'],
                exploration_final_eps=self.parameters['exploration_final_eps'],
                exploration_fraction=self.parameters['exploration_fraction'],
                train_freq=self.parameters['train_freq'],
                gamma=self.parameters['gamma'],
                policy_kwargs=policy_kwargs
                )
        callback = Callback(
            self.agent, 
            self.original_env,
            self.eval_env,
            self._run,
            deterministic=self.parameters['eval_deterministic'],
            eval_freq=self.parameters['eval_freq'],
            n_eval_episodes=self.parameters['eval_episodes'],
            log_dir=self.log_dir
            )
        self.agent.learn(total_timesteps=self.parameters['total_steps'], reset_num_timesteps=False, callback=callback)
        

if __name__ == '__main__':
    ex = sacred.Experiment('policy_confounding')
    ex.add_config('configs/default.yaml')
    add_mongodb_observer()

    @ex.automain
    def main(parameters, seed, _run):
        exp = Experiment(parameters, _run, seed)
        exp.learn()
