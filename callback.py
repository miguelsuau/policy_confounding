import re
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# from sensitivity import Sensitivity

class Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, agent, train_env, eval_env, sacred_run, deterministic, eval_freq, n_eval_episodes, log_dir, verbose=0):
        super(Callback, self).__init__(verbose)
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.sacred_run = sacred_run
        self.n_calls = 0
        self.deterministic = deterministic
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        # self.sensitivity = Sensitivity(self.agent, self.train_env)
        
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self._try_policy_train_eval_envs()
        # relative_entropy = self.sensitivity.measure()
        # self.sacred_run.log_scalar('relative_entroypy', np.mean(relative_entropy[1]), self.n_calls)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.eval_freq == 0:
            self._try_policy_train_eval_envs()
            # relative_entropy = self.sensitivity.measure()
            # print(relative_entropy)
            # self.sacred_run.log_scalar('relative_entroypy', np.mean(relative_entropy[1]), self.n_calls)
            # breakpoint()
        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    
    def _try_policy_train_eval_envs(self):
        rewards_train, length_train = evaluate_policy(
                self.agent, 
                self.train_env, 
                n_eval_episodes=self.n_eval_episodes, 
                deterministic=self.deterministic,
                return_episode_rewards=True
                )
        mean_reward_train = np.mean(rewards_train)
        mean_length_train = np.mean(length_train)
        print('Train step: %i' % self.n_calls)
        print(("-"*80))
        print('Training environment - Episode mean reward: %.2f, Episode mean length: %.2f' % (mean_reward_train, mean_length_train))
        self.sacred_run.log_scalar('Train mean reward', mean_reward_train, self.n_calls)

        rewards_eval, length_eval = evaluate_policy(
            self.agent, 
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            render=False
            )
        mean_reward_eval = np.mean(rewards_eval)
        mean_length_eval = np.mean(length_eval)
        print('Evaluation environment - Episode mean reward: %.2f, Episode mean length: %.2f' % (mean_reward_eval, mean_length_eval))
        print(("-"*80))
        self.sacred_run.log_scalar('Eval mean reward', mean_reward_eval, self.n_calls)
        
