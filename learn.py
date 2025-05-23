from logging import INFO
import numpy as np
import os
import sys
sys.path.append("..")

import gym
from copy import deepcopy

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from custom_ppo import customPPO

from callback import Callback
from stable_baselines3.common.buffers import ReplayBuffer

from reinforce import REINFORCE

import mlflow
import argparse
import yaml


class Experiment(object):
    """
    Creates experiment object to interact with the environment and
    the agent and log results.
    """
    def __init__(self, parameters, seed):
        self._seed = seed
        self.parameters = parameters
        self.seed = seed
        self.log_dir = 'tmp/'
        os.makedirs(self.log_dir, exist_ok=True)
        self.create_env()

    def create_env(self):
        
        env_id = self.parameters['name']
        
        env = DummyVecEnv([lambda: gym.make(
            env_id, 
            seed=np.random.randint(1.0e+6), 
            eval=False
            )])
        if self.parameters['n_stack'] > 1:
            env = VecFrameStack(env, n_stack=self.parameters['n_stack'])
        self.original_env = VecMonitor(env, self.log_dir)

        env = DummyVecEnv([lambda: gym.make(
            env_id, 
            seed=np.random.randint(1.0e+6),
            eval=False,
            random_action_prob=self.parameters['random_action_prob']
            )])

        if self.parameters['n_stack'] > 1:
            env = VecFrameStack(env, n_stack=self.parameters['n_stack'])
        self.train_env = VecMonitor(env, self.log_dir)
        

        env = DummyVecEnv([lambda: gym.make(
            env_id, 
            seed=np.random.randint(1.0e+6), 
            eval=True
            )])
        if self.parameters['n_stack'] > 1:
            env = VecFrameStack(env, n_stack=self.parameters['n_stack'])
        self.eval_env = VecMonitor(env, self.log_dir)
    
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
            self.agent = customPPO(
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
                gae_lambda=self.parameters['gae_lambda'],
                learning_rate=self.linear_schedule(self.parameters['learning_rate'], self.parameters['learning_rate_final']),
                gamma=self.parameters['gamma'],
                policy_kwargs=policy_kwargs,
                normalize_advantage=self.parameters['normalize_advantage'],
                use_advantage=self.parameters['use_advantage']
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
                target_update_interval=self.parameters['target_update_interval'],
                exploration_initial_eps=self.parameters['exploration_initial_eps'],
                exploration_final_eps=self.parameters['exploration_final_eps'],
                exploration_fraction=self.parameters['exploration_fraction'],
                train_freq=self.parameters['train_freq'],
                gamma=self.parameters['gamma'],
                policy_kwargs=policy_kwargs
            )
        
        elif self.parameters['algorithm'] == 'REINFORCE':
            self.agent = REINFORCE(
                self.train_env,
                learning_rate=self.parameters['learning_rate'],
                gamma=self.parameters['gamma'],
                use_advantage=self.parameters['use_advantage'],
                beta=self.parameters['beta']
                )
            
        callback = Callback(
            self.agent, 
            self.original_env,
            self.train_env,
            self.eval_env,
            deterministic=self.parameters['eval_deterministic'],
            eval_freq=self.parameters['eval_freq'],
            n_eval_episodes=self.parameters['eval_episodes'],
            log_dir=self.log_dir
            )
        self.agent.learn(total_timesteps=self.parameters['total_steps'], reset_num_timesteps=False, callback=callback)

def merge_configs(default_config, custom_config):
    """
    Merges the custom config into the default config.
    If a key exists in both, the custom config overrides the default.
    """
    merged_config = deepcopy(default_config)
    merged_config.update(custom_config)
    return merged_config    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run RL experiment with MLFlow")
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--default_config', type=str, default='configs/default.yaml',
                        help='Path to the default configuration file')

    # Parse arguments
    args = parser.parse_args()

    # Load default configuration
    default_config_path = args.default_config
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")
    
    with open(default_config_path, 'r') as default_config_file:
        default_parameters = yaml.load(default_config_file, Loader=yaml.FullLoader)

    # Load custom configuration (if provided)
    custom_config_path = args.config
    if not os.path.exists(custom_config_path):
        raise FileNotFoundError(f"Custom config file not found: {custom_config_path}")

    with open(custom_config_path, 'r') as custom_config_file:
        custom_parameters = yaml.load(custom_config_file, Loader=yaml.FullLoader)

    # Merge configurations (custom overrides default)
    parameters = merge_configs(default_parameters['parameters']['main'], custom_parameters['parameters'].get('main', {}))

    # Set up MLFlow experiment
    mlflow.set_experiment(f'policy_confounding_{parameters["env"]}')
    
    with mlflow.start_run():
        # You can log parameters at the start of the experiment
        mlflow.log_params(parameters)
        
        # Seed tracking as well
        seed = 42  # or dynamically generate
        mlflow.log_param("seed", seed)

        exp = Experiment(parameters, seed)
        exp.learn()