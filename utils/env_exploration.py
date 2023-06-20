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
            return torch.tensor([[self._env.action_space.sample()]], dtype=torch.long).to(self._ctx[0])


class Gaussian(EnvExplorer):
    def __init__(self, opts, ctx, action_delegate):
        super().__init__(opts, ctx, action_delegate)
        self._opts = opts
        self._mu = 0
        self._sigma = 1

    def explore(self, state, global_iter, **kwargs):
        action = self._select_action(state)
        noise = torch.normal(self._mu, self._sigma, size=action.shape).to(action.device)
        return action + noise.clip(-self._opts.max_action, self._opts.max_action)


class OrnsteinUlhenbeckNoise:
    """Ornstein-Uhlenbeck process.

    The OU_Noise class has four attributes

        size: the size of the noise vector to be generated
        mu: the mean of the noise, set to 0 by default
        theta: the rate of mean reversion, controlling how quickly the noise returns to the mean
        sigma: the volatility of the noise, controlling the magnitude of fluctuations
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.25):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample.
        This method uses the current state of the noise and generates the next sample
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state
