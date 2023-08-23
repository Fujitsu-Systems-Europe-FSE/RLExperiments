from typing import List
from torchvision import transforms
from collections import namedtuple, deque

import torch
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transforms = None

    def add_transforms(self, trans: List):
        self.transforms = transforms.Compose(trans)

    def push(self, *args):
        """Save a transition"""
        data = []
        for e in args:
            if type(e) == torch.Tensor:
                data.append(e.cpu().numpy())
            elif e.dtype == np.float64:
                data.append(e.astype(np.float32))
            else:
                data.append(e)

        # split vectorized data
        data = [np.split(t, t.shape[0], axis=0) for t in data]
        for s, a, n, r, d in zip(*data):
            self.memory.append(Transition(s, a, n, r, d))

    def sample(self, batch_size):
        transitions_list = random.sample(self.memory, batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # batch = Transition(*zip(*transitions_list))
        states_list, actions_list, next_states_list, rewards_list, dones_list = map(tuple, zip(*transitions_list))

        states = np.concatenate(states_list, axis=0)
        next_states = np.concatenate(next_states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        rewards = np.stack(rewards_list, axis=0)
        dones = np.stack(dones_list, axis=0)

        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        actions = torch.tensor(actions)#.float()
        rewards = torch.tensor(rewards)#.float()
        dones = torch.tensor(dones).int()

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.memory)
