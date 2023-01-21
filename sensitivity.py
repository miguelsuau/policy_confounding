from scipy.special import rel_entr
import torch
import copy
import numpy as np
class Sensitivity:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def measure(self):
        
        if self.env.get_attr('NAME')[0] == 'tmaze':
            relative_entropy = self.tmaze_sensitivity()
        return relative_entropy
        

    def tmaze_sensitivity(self):
        optimal_policy = self.env.get_attr('optimal_policy')[0]
        relative_entropy = {}
        # self.env.set_attr('FIXED_VALUE', True, 0)
        for v in [-1, 1]: 
            # self.env.set_attr('value', v, 0)
            self.env.env_method('set_fixed_value', value=v)
            # self.env.env_method('set_fixed_start')
            obs = self.env.reset()
            # print(self.env.get_attr('value'))
            action_vector = optimal_policy(v)
            relative_entropy[v] = []
            # for i, action in enumerate(action_vector):
            done = False
            while not done:
                action, _ = self.agent.predict(obs)
                # print(self.env.get_attr('location'), action)
                dist_1 = self.agent.policy.get_distribution(torch.FloatTensor(obs))
                # print(np.reshape(obs, (self.env.n_stack, -1)))
                probs_1 = dist_1.distribution.probs.detach().numpy()
                
                # change value index
                obs_2 = copy.deepcopy(obs)
                obs_2 = np.reshape(obs_2, (self.env.n_stack, -1))
                obs_2[:,-1][obs_2[:,-1] != 0] = 1 if v == -1 else -1
                # print(obs_2)
                # breakpoint()
                dist_2 = self.agent.policy.get_distribution(torch.FloatTensor(np.reshape(obs_2, (1,-1))))
                probs_2 = dist_2.distribution.probs.detach().numpy()
                # print(probs_1, probs_2)
                relative_entropy[v].append(sum(rel_entr(probs_1[0], probs_2[0])))

                obs, _, done, _ = self.env.step([action])        
        self.env.env_method('set_random_value')
        # self.env.env_method('set_random_start')
        return relative_entropy