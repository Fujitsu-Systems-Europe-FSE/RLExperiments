from typing import List
from torchvision import transforms
from collections import namedtuple, deque

import torch
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transforms = None

    def add_transforms(self, trans: List):
        self.transforms = transforms.Compose(trans)

    def push(self, *args):
        """Save a transition"""
        # assert np.all([t.device.type == 'cpu' for t in args]), 'Tensors must be moved to CPU'
        data = []
        for e in args:
            if type(e) == torch.Tensor:
                e = e.cpu()
            data.append(e)
        self.memory.append(Transition(*data))

    def sample(self, batch_size):
        transitions_list = random.sample(self.memory, batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # batch = Transition(*zip(*transitions_list))
        states_list, actions_list, next_states_list, rewards_list = map(tuple, zip(*transitions_list))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_states = list(map(lambda s: s is not None, next_states_list))
        non_final_masks = torch.tensor(non_final_states)
        non_final_next_states = torch.cat([s for s in next_states_list if s is not None])

        states = torch.cat(states_list, dim=0)
        actions = torch.cat(actions_list, dim=0)
        rewards = torch.cat(rewards_list, dim=0)

        next_states = torch.zeros((len(non_final_states), *non_final_next_states.shape[1:]))
        next_states[non_final_masks] = non_final_next_states

        return states, actions, next_states, rewards, non_final_masks.unsqueeze(-1)

    def __len__(self):
        return len(self.memory)
