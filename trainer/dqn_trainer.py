from time import time
from itertools import count
from dataset.replay_memory import Transition
from apheleia.trainer.rl_trainer import RLTrainer
from apheleia.metrics.metric_store import MetricStore
from apheleia.metrics.average_meter import AverageMeter

import math
import torch
import random


class DQNTrainer(RLTrainer):

    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, *args, **kwargs):
        super().__init__(opts, net, optims, scheds, loss, validator, metrics, ctx, 'RL', *args, **kwargs)
        self._gamma = 0.99
        self._eps_start = 0.9
        self._eps_end = 0.05
        self._eps_decay = 1000
        self._tau = 0.005

        metrics.add_train_metric('episodes/duration_in_steps', AverageMeter(''))
        metrics.add_train_metric('episodes/duration_in_secs', AverageMeter(''))

    def _train_loop(self, memory, *args, **kwargs):
        # Initialize the environment and get it's state
        state, info = self._environment.reset()
        state = torch.tensor(state).unsqueeze(0).to(self._ctx[0])

        t0 = time()
        for t in count():
            action = self._select_action(state)
            observation, reward, terminated, truncated, _ = self._environment.step(action.item())
            reward = torch.tensor([reward]).to(self._ctx[0])
            done = terminated or truncated
            next_state = None if terminated else torch.tensor(observation).to(self._ctx[0]).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self._optimize(memory)
            self._soft_update()

            if done:
                self._metrics_store.update_train({
                    'episodes/duration_in_steps': t + 1,
                    'episodes/duration_in_secs': time() - t0
                })
                break

    def _soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self._net['TargetNet'].state_dict()
        policy_net_state_dict = self._net['PolicyNet'].state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self._tau + target_net_state_dict[key] * (1 - self._tau)
        self._net['TargetNet'].load_state_dict(target_net_state_dict)

    def _optimize(self, memory):
        if len(memory) < self._opts.batch_size:
            return

        action, next_state, reward, state = memory.sample(self._opts.batch_size)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_states = list(map(lambda s: s is not None, next_state))
        non_final_mask = torch.tensor(non_final_states).to(self._ctx[0])
        non_final_next_states = torch.cat([s for s in next_state if s is not None])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        with torch.cuda.amp.autocast(self._opts.fp16):
            state_action_values = self._net['PolicyNet'](torch.cat(state, dim=0)).gather(1, torch.cat(action, dim=0))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_action_values = torch.zeros(self._opts.batch_size).to(self._ctx[0])
        with torch.no_grad():
            estimated_rewards = self._net['TargetNet'](non_final_next_states)
            max_rewards, max_indexes = estimated_rewards.max(dim=1)
            next_state_action_values[non_final_mask] = max_rewards
        # Compute the expected Q values
        # r + gamma * Q(s', a)
        expected_next_state_action_values = torch.cat(reward, dim=0) + self._gamma * next_state_action_values

        # Compute temporal difference error
        loss = self._loss.compute(state_action_values, expected_next_state_action_values.unsqueeze(1))

        # Same optimizer used for all models
        self._optimizer['PolicyNet'].zero_grad()
        self._scaler.scale(loss).backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._net['PolicyNet'].parameters(), 100)
        self._scaler.step(self._optimizer['PolicyNet'])
        self._scaler.update()

        # self._log_iteration(batch_idx)
        self._metrics_store.update_train(self._loss.decompose())

    def _select_action(self, state):
        # Here we use an epsilon greedy policy.
        # Sometimes we use our model for choosing an action. Sometimes we are just sampling a random one
        sample = random.random()
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * self.global_iter / self._eps_decay)
        self.global_iter += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Pick action with the larger expected reward.
                # argmax Q(s, a)
                estimated_rewards = self._net['PolicyNet'](state)
                max_reward, max_index = estimated_rewards.max(dim=1)
                return max_index.view(1, 1)
        else:
            return torch.tensor([[self._environment.action_space.sample()]], dtype=torch.long).to(self._ctx[0])

    def _save_checkpoints(self, out_filename):
        pass

    def _report_stats(self, *args):
        pass

    def get_graph(self) -> torch.nn.Module:
        pass
