import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SharedNetwork(nn.Module):
    def __init__(self, observation_space, action_space, lr=3e-4):
        super(SharedNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Policy head (produces logits)
        self.policy_head = nn.Linear(128, action_space.n)
        
        # Value head (produces state-value estimate V(s))
        self.value_head = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs):
        features = self.shared_net(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)  # Ensure output shape is (batch,)
        return logits, value

    def predict(self, obs, deterministic=False):
        logits, _ = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample() if not deterministic else torch.argmax(probs)
    
    def action_prob(self, obs):
        logits, _ = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        return probs


class REINFORCE:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, use_advantage=True, beta=0.01):
        self.env = env
        self.gamma = gamma
        self.network = SharedNetwork(env.observation_space, env.action_space, lr=learning_rate)
        self.optimizer = self.network.optimizer
        self.num_timesteps = 0  # Track training timesteps
        self.use_advantage = use_advantage
        self.beta = beta

    # def compute_returns(self, rewards, values):
    #     """Estimate discounted returns using rewards and value estimates"""
    #     returns = []
    #     values = [value.detach().item() for value in values]
    #     values.append(0)  # Add a zero to the end for the last state
    #     values = values[1:]  # Discard the first value estimate
    #     for reward, value in zip(rewards,values):
    #         returns.append([reward[0] + self.gamma * value])

        # return torch.tensor(returns, dtype=torch.float32)

    def compute_returns(self, rewards):
        """Compute discounted returns for REINFORCE"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(np.array(returns), dtype=torch.float32)

    def collect_trajectory(self, callback=None):
        """Run an episode and collect trajectory data"""
        obs = self.env.reset()
        done = False
        log_probs = []
        rewards = []
        values = []
        states = []
        entropies = []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits, value = self.network(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            entropy = action_dist.entropy()
            
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            next_obs, reward, done, _ = self.env.step([action.item()])
            
            # Callback step
            if callback is not None:
                callback.update_locals(locals())
                callback.on_step()

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            states.append(obs_tensor)
            entropies.append(entropy)
            obs = next_obs

            self.num_timesteps += 1  # Track timesteps

        return log_probs, rewards, values, states, entropies

    def train(self, callback=None):
        """Train the policy using REINFORCE with baseline"""
        all_log_probs = []
        all_rewards = []
        all_values = []
        all_states = []
        all_entropies = []
        
        # Collect 'batch_size' steps
    
        log_probs, rewards, values, states, entropies = self.collect_trajectory(callback)
        all_log_probs.append(log_probs)
        all_rewards.append(rewards)
        all_values.append(values)
        all_states.append(states)
        all_entropies.append(entropies)

        # Compute returns for all episodes in batch
        # all_returns = [self.compute_returns(rewards, values) for rewards, values in zip(all_rewards, all_values)]
        all_returns = [self.compute_returns(rewards) for rewards in all_rewards]

        # Flatten log_probs, values, and returns
        log_probs_flat = torch.cat([torch.stack(lp) for lp in all_log_probs])
        values_flat = torch.cat([torch.stack(v) for v in all_values])
        returns_flat = torch.cat(all_returns)
        entropies_flat = torch.cat([torch.stack(e) for e in all_entropies])
        # Compute advantage function: A = G - V(s)
        if self.use_advantage:
            advantages = returns_flat - values_flat.detach()
        else:
            advantages = returns_flat
        
        # Policy loss using advantage function
        policy_loss = -torch.mean(log_probs_flat * advantages)

        # Value function loss (MSE between V(s) and G)
        value_loss = torch.nn.functional.mse_loss(values_flat, returns_flat)


        # Update policy and value function together
        self.optimizer.zero_grad()
        (policy_loss + value_loss - self.beta * entropies_flat.mean()).backward()
        self.optimizer.step()

    def learn(self, total_timesteps, callback=None, reset_num_timesteps=True):
        """Train for a given number of timesteps while using callbacks"""
        self.num_timesteps = 0  # Reset timesteps

        if callback is not None:
            callback.init_callback(self)  # Initialize the callback

        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            self.train(callback)

        # Callback finish
        if callback is not None:
            callback.on_training_end()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Predict an action based on the current policy"""
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        return [self.network.predict(obs_tensor, deterministic).item()], None


# Example Callback (SB3-compatible)
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Called at each step of training"""
        if self.n_calls % 10 == 0:
            print(f"Callback: Step {self.n_calls}, Total Timesteps: {self.model.num_timesteps}")
        return False  # Returning True would stop training

    def _on_training_end(self):
        """Called when training ends"""
        print("Training finished!")


