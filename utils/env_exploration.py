from abc import abstractmethod, ABCMeta

import copy
import math
import torch
import random
import numpy as np


class EnvExplorer(metaclass=ABCMeta):

    def __init__(self, opts, ctx, action_delegate):
        self._opts = opts
        self._ctx = ctx
        self._env = opts.env
        self._select_action = action_delegate

    @abstractmethod
    def explore(self, state, global_iter, *args, **kwargs):
        pass


class EpsilonGreedy(EnvExplorer):
    def __init__(self, opts, ctx, action_delegate):
        super().__init__(opts, ctx, action_delegate)
        self._eps_start = 0.9
        self._eps_end = 0.05
        self._eps_decay = 1000

    def explore(self, state, global_iter, **kwargs):
        sample = random.random()
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * global_iter / self._eps_decay)
        if sample > eps_threshold:
            return self._select_action(state)
        else:
            sample = self._env.action_space.sample()
            sample = torch.tensor([[sample]], dtype=torch.long) if type(sample) == np.int64 else sample['continuous']
            return sample.to(self._ctx[0])


class Gaussian(EnvExplorer):
    def __init__(self, opts, ctx, action_delegate):
        super().__init__(opts, ctx, action_delegate)
        self._opts = opts
        self._mu = 0
        self._sigma = 1

    def explore(self, state, global_iter, **kwargs):
        action = self._select_action(state)
        noise = torch.normal(self._mu, self._sigma, size=action.shape).to(action.device)
        mini = torch.tensor(self._opts.min_actions).to(action.device)
        maxi = torch.tensor(self._opts.max_actions).to(action.device)
        return (action + noise).clamp(mini, maxi)


class OrnsteinUlhenbeck(EnvExplorer):
    def __init__(self, opts, ctx, action_delegate, mu=0., sigma=.2, theta=0.15, dt=1e-2, initial_noise=None):
        super().__init__(opts, ctx, action_delegate)

        self.theta = theta
        self.mu = mu * np.ones((1, 1))  # mean
        self.sigma = sigma * np.ones((1, 1))   # std
        self.dt = dt
        self.initial_noise = initial_noise

        self.reset()

    def explore(self, state, global_iter, *args, **kwargs):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        noise = (
            self.prev_noise
            + self.theta * (self.mu - self.prev_noise) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.prev_noise = noise

        action = self._select_action(state)
        noise = torch.tensor(noise, dtype=torch.float32).to(action.device)
        mini = torch.tensor(self._opts.min_actions).to(action.device)
        maxi = torch.tensor(self._opts.max_actions).to(action.device)
        return (noise + action).clamp(mini, maxi)

    def reset(self):
        if self.initial_noise is not None:
            self.prev_noise = self.initial_noise
        else:
            self.prev_noise = np.zeros_like(self.mu)


# class OrnsteinUlhenbeckNoise:
#     """Ornstein-Uhlenbeck process.
#
#     The OU_Noise class has four attributes
#
#         size: the size of the noise vector to be generated
#         mu: the mean of the noise, set to 0 by default
#         theta: the rate of mean reversion, controlling how quickly the noise returns to the mean
#         sigma: the volatility of the noise, controlling the magnitude of fluctuations
#     """
#     def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.25):
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.seed = random.seed(seed)
#         self.reset()
#
#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = copy.copy(self.mu)
#
#     def sample(self):
#         """Update internal state and return it as a noise sample.
#         This method uses the current state of the noise and generates the next sample
#         """
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
#         self.state += dx
#         return self.state
