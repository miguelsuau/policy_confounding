import re
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import mlflow
import torch
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from sensitivity import Sensitivity

class Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, agent, original_env, train_env, eval_env, deterministic, eval_freq, n_eval_episodes, log_dir, verbose=0):
        super(Callback, self).__init__(verbose)
        self.agent = agent
        self.env = original_env
        self.train_env = train_env
        self.eval_env = eval_env
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
        # self._log_kl_divergence()
        if self.agent.__class__.__name__ != 'REINFORCE':
            self.generate_heatmap()
        # self.estimate_probs_and_advantages()
        self.i = 0
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
        call_freq = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        reward_freq = 1000

        if self.n_calls % self.eval_freq == 0:
            self._try_policy_train_eval_envs()
            # self._log_kl_divergence()
            if self.agent.__class__.__name__ != 'REINFORCE':
                self.generate_heatmap()
            # relative_entropy = self.sensitivity.measure()
            # print(relative_entropy)
        # if self.n_calls % (call_freq[self.i] + 1) == 0:
        #     if len(call_freq) > self.i + 1:
        #         self.i += 1
        #     self.estimate_probs_and_advantages()
        # if self.n_calls % reward_freq == 0:
        #     self._get_n_rewards(reward_freq)
            # self.sacred_run.log_scalar('relative_entroypy', np.mean(relative_entropy[1]), self.n_calls)
            # breakpoint()
        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        mlflow.log_metric('advantage_norm', np.mean(np.abs(self.agent.rollout_buffer.advantages)), self.n_calls)
        
        returns = np.zeros_like(self.agent.rollout_buffer.rewards)
        last_return = 0
        for step in reversed(range(self.agent.rollout_buffer.buffer_size)):
            if step == self.agent.rollout_buffer.buffer_size - 1:
                next_non_terminal = 0
                last_return = 0
            else:
                next_non_terminal = 1.0 - self.agent.rollout_buffer.episode_starts[step + 1]
            last_return =  self.agent.rollout_buffer.rewards[step] + self.agent.rollout_buffer.gamma * last_return * next_non_terminal
            returns[step] = last_return
        
        # find last episode starts
        last_episode_start = np.where(self.agent.rollout_buffer.episode_starts == 1)[0][-1]
        # remove incomplete episodes
        returns = returns[:last_episode_start]
        
        value_mse = np.mean((self.agent.rollout_buffer.values[:last_episode_start] - returns) ** 2)
        mlflow.log_metric('value_mse', value_mse, self.n_calls)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    
    def _try_policy_train_eval_envs(self, steps=0):
        rewards_train, length_train = evaluate_policy(
                self.agent, 
                self.env, 
                n_eval_episodes=self.n_eval_episodes, 
                deterministic=self.deterministic,
                return_episode_rewards=True
                )
        mean_reward_train = np.mean(rewards_train)
        mean_length_train = np.mean(length_train)
        print('Train step: %i' % self.n_calls)
        print(("-"*80))
        print('Training environment - Episode mean reward: %.2f, Episode mean length: %.2f' % (mean_reward_train, mean_length_train))
        mlflow.log_metric('Train mean reward', mean_reward_train, self.n_calls)
        # self.sacred_run.log_scalar('Train mean reward', mean_reward_train, self.n_calls)

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
        mlflow.log_metric('Eval mean reward', mean_reward_eval, self.n_calls)
        # self.sacred_run.log_scalar('Eval mean reward', mean_reward_eval, self.n_calls)

    def generate_random_trajectory(self, env, n_steps=10):
        obs = env.reset()
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step([action])
            if done:
                break
        
    def _log_kl_divergence(self):
        obs1, obs2, wrong_action = self.env.env_method("get_test_obs")[0]
        obs1 = torch.tensor(obs1).unsqueeze(0)
        obs2 = torch.tensor(obs2).unsqueeze(0)
        log_probs1, log_probs2, kl_divergence = self._measure_policy_kl_divergence(obs1, obs2)

        prob_wrong_action = log_probs2.exp()[wrong_action].item()
        print('KL Divergence: %.2f' % kl_divergence)
        print(("-"*80))
        print("Prob wrong action: %.2f" % prob_wrong_action)
        print(("-"*80))
        mlflow.log_metric('KL Divergence', kl_divergence, self.n_calls)
        mlflow.log_metric('Prob wrong action', prob_wrong_action, self.n_calls)

    def _measure_policy_kl_divergence(self, obs1, obs2):
        actions = torch.tensor(list(self.env.get_attr("ACTIONS")[0].keys()))
        _, log_probs1, _, = self.agent.policy.evaluate_actions(obs1, actions)
        _, log_probs2, _, = self.agent.policy.evaluate_actions(obs2, actions)
        kl_divergence = (log_probs1.exp() * (log_probs1 - log_probs2)).sum().item()
        return log_probs1, log_probs2, kl_divergence

    def generate_heatmap(self, n_episodes=100):
        if self.env.get_attr("NAME")[0] == 'keydoor':
            corridor_length = self.env.get_attr("CORRIDOR_LENGTH")[0]
            heatmap = np.zeros(corridor_length - 1)
            counts = np.zeros(corridor_length - 1)

            obs1 = self.env.reset()
            obs2 = obs1.copy()
            obs2[0, -1] = 1 if obs1[0, -1] == 0 else 0
            obs1 = torch.tensor(obs1).unsqueeze(0)
            obs2 = torch.tensor(obs2).unsqueeze(0)

            for _ in range(n_episodes):
                done = False
                while not done:
                    _, _, divergence = self._measure_policy_kl_divergence(obs1, obs2)
                    location = self.env.get_attr("location")[0]
                    heatmap[location] += divergence
                    counts[location] += 1
                    action = self.agent.policy(obs1)[0]
                    obs1, reward, done, info = self.env.step([action])
                    obs2 = obs1.copy()
                    obs2[0, -1] = 1 if obs1[0, -1] == 0 else 0
                    obs1 = torch.tensor(obs1).unsqueeze(0)
                    obs2 = torch.tensor(obs2).unsqueeze(0)
            counts = np.maximum(counts, 1)
            heatmap = heatmap / counts
            heatmap = heatmap.reshape(1, -1)

        elif self.env.get_attr("NAME")[0] == 'tmaze':
            obs1 = self.env.reset()
            obs1 = torch.tensor(obs1).unsqueeze(0)
            corridor_length = self.env.get_attr("CORRIDOR_LENGTH")[0]
            corridor_width = self.env.get_attr("CORRIDOR_WIDTH")[0]
            heatmap = np.zeros((corridor_width, corridor_length))
            counts = np.zeros((corridor_width, corridor_length))
            for _ in range(n_episodes):
                done = False
                while not done:
                    action = self.agent.policy(obs1)[0]
                    obs1, reward, done, info = self.env.step([action])
                    obs2 = obs1.copy()
                    # get index of every 10th element in obs1
                    values = obs1[0,corridor_length*corridor_width::corridor_length*corridor_width+1]
                    # replace 1 by -1 and -1 by 1
                    if self.env.get_attr("signal")[0] == 1:
                        values = np.where(values == 1, -1, values)
                    else:
                        values = np.where(values == -1, 1, values)
                    obs2[0,corridor_length*corridor_width::corridor_length*corridor_width+1] = values
                    obs1 = torch.tensor(obs1).unsqueeze(0)
                    obs2 = torch.tensor(obs2).unsqueeze(0)
                    _, _, divergence = self._measure_policy_kl_divergence(obs1, obs2)
                    location = self.env.get_attr("location")[0]
                    heatmap[location[0], location[1]] += divergence
                    counts[location[0], location[1]] += 1
            # counts = np.maximum(counts, 1)
            heatmap = heatmap / counts
            # where nan values are present, replace them with 0
            # heatmap = np.nan_to_num(heatmap, nan=-1)
        elif self.env.get_attr("NAME")[0] == 'diversion':
            obs1 = self.env.reset()
            obs1 = torch.tensor(obs1).unsqueeze(0)
            corridor_length = self.env.get_attr("CORRIDOR_LENGTH")[0]
            heatmap = np.zeros(corridor_length - 1)
            counts = np.zeros(corridor_length - 1)
            for _ in range(n_episodes):
                done = False
                while not done:
                    action = self.agent.policy(obs1)[0]
                    obs1, reward, done, info = self.env.step([action])
                    obs2 = obs1.copy()
                    obs2[0, -1] = 1 if obs1[0, -1] == 0 else 0
                    obs1 = torch.tensor(obs1).unsqueeze(0)
                    obs2 = torch.tensor(obs2).unsqueeze(0)
                    _, _, divergence = self._measure_policy_kl_divergence(obs1, obs2)
                    location = self.env.get_attr("location")[0][1]
                    heatmap[location] += divergence
                    counts[location] += 1
            counts = np.maximum(counts, 1)
            heatmap = heatmap / counts
            heatmap = heatmap.reshape(1, -1)
                    
        else:
            return None

        # plot heatmap
        fig, ax = plt.subplots()  # Create a new figure and axis

        # Plot the heatmap with grid lines
        cax = ax.imshow(heatmap, vmin=0, vmax=3, cmap='viridis')  # Customize the colormap if needed
        # if value in heatmap is -1 then make heatmap value white
        cax.cmap.set_bad(color='white')
        # Add white grid lines
        ax.set_xticks(np.arange(0.5, heatmap.shape[1] - 0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, heatmap.shape[0] - 0.5, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which="minor", size=0)  # Hide minor ticks
        
        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
         # Create a horizontal colorbar below the heatmap
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("bottom", size="10%", pad=0.4)  # Adjust size and padding
        fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
            

        # Save the heatmap
        path = f'heatmaps/{self.env.get_attr("NAME")[0]}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_path = f'{path}/heatmap_{self.n_calls}.png'
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to release memory
        
        mlflow.log_image(heatmap, save_path)
        return heatmap

    def estimate_probs_and_advantages(self, n_episodes=2000):

        obs = self.env.reset()
        obs = torch.tensor(obs).unsqueeze(0)

        counts_negative_reward = 0
        counts_positive_reward = 0

        location_value = 0
        location_key = 0
        location_5_counts = 0

        for _ in range(n_episodes):
            done = False
            while not done:
                location = self.env.get_attr("location")[0]
                action = self.agent.policy(obs)[0]
                has_key = self.env.get_attr("has_key")[0]
                obs, reward, done, info = self.env.step([action])
                if done:
                    if reward == 1:
                        counts_positive_reward += 1
                    else:
                        counts_negative_reward += 1
                    if location == 5:
                        location_5_counts += 1
                        location_value += reward
                        location_key += has_key
                    

                obs = torch.tensor(obs).unsqueeze(0)
                    
        mlflow.log_metric('Percentage Negative Reward', counts_negative_reward/n_episodes, self.n_calls)
        mlflow.log_metric('Percentage Positive Reward', counts_positive_reward/n_episodes, self.n_calls)
        obs1, obs2, wrong_action = self.env.env_method("get_test_obs")[0]
        obs1 = torch.tensor(obs1).unsqueeze(0)
        obs2 = torch.tensor(obs2).unsqueeze(0)
        value1 = self.agent.policy(obs1)[1]
        value2 = self.agent.policy(obs2)[1]
        positive_reward = self.env.get_attr("POSITIVE_REWARD")[0]
        negative_reward = self.env.get_attr("NEGATIVE_REWARD")[0]
        advantage_positive_reward = abs(positive_reward - value1)
        advantage_negative_reward = abs(negative_reward - value2)
        mlflow.log_metric('Advantage Positive Reward', advantage_positive_reward, self.n_calls)
        mlflow.log_metric('Advantage Negative Reward', advantage_negative_reward, self.n_calls)
        mlflow.log_metric('Location 6 Value', location_value/location_5_counts, self.n_calls)
        mlflow.log_metric('Location 6 Key', location_key/location_5_counts, self.n_calls)


    def _get_n_rewards(self, freq):
        n_positive_rewards = self.train_env.get_attr("n_positive_rewards")[0]
        n_negative_rewards = self.train_env.get_attr("n_negative_rewards")[0]
        mlflow.log_metric("Number Negative Reward", n_negative_rewards, self.n_calls)
        mlflow.log_metric("Number Positve Reward", n_positive_rewards, self.n_calls)
