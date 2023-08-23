from torch import nn

from apheleia import ProjectLogger
from apheleia.utils import to_tensor
from utils.env_exploration import EpsilonGreedy
from apheleia.metrics.meters import AverageMeter
from apheleia.trainer.rl_trainer import RLTrainer
from apheleia.metrics.metric_store import MetricStore
from apheleia.utils.visualize import gradients_norm_hist
from apheleia.utils.gradients import calc_jacobian_norm, calc_net_gradient_norm

import torch
import numpy as np


class DQNTrainer(RLTrainer):

    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, *args, **kwargs):
        super().__init__(opts, net, optims, scheds, loss, validator, metrics, ctx, 'RL', *args, **kwargs)
        self._add_metrics()
        self._initial_setup()
        self._state = ...

    def _initial_setup(self):
        self._env_explorer = EpsilonGreedy(self._opts, self._ctx, self._select_action)
        # TODO Move in ModelStore ?
        clazz = self._net['PolicyNet'].module.__class__
        self._target_net = self._targetize_net(clazz)

    def _targetize_net(self, netclazz):
        net = netclazz(self._opts)
        target_net = nn.DataParallel(net, device_ids=self._ctx).to(self._ctx[0])
        target_net.load_state_dict(self._net.get(net.model_name()).state_dict())
        return target_net

    def _add_metrics(self):
        reference_metric = AverageMeter('')
        self._metrics_store.add_target_metric(reference_metric)
        self._metrics_store.add_train_metric('rewards/episodic', reference_metric)
        self._metrics_store.add_train_metric('episodes/duration_in_steps', AverageMeter(''))
        self._metrics_store.add_train_metric('episodes/duration_in_secs', AverageMeter(''))

    def _pre_loop_hook(self, memory, *args):
        # https://gymnasium.farama.org/api/vector/
        # Vectorized environments are automatically reset
        # Initialize the environment and get it's state
        if self.global_iter == 0:
            try:
                self._state, _ = self._environment.reset(seed=self._opts.seed)
            except Exception as e:
                ProjectLogger().warning(f'Failed to seed environment : {e}')
                self._state, _ = self._environment.reset()
            self._state = self._state if memory.transforms is None else memory.transforms(self._state)

    def _train_loop(self, memory, *args, **kwargs):
        action = self._env_explorer.explore(self._state, self.global_iter)
        next_state, reward, terminated, truncated, infos = self._environment.step(action)
        next_state = next_state if memory.transforms is None else memory.transforms(next_state)

        if 'final_info' in infos:
            for info in infos['final_info']:
                if info is None:
                    continue
                self._metrics_store.update_train({
                    'rewards/episodic': info['episode']['r'].item(),
                    'episodes/duration_in_steps': info['episode']['l'].item(),
                    'episodes/duration_in_secs': info['episode']['t'].item()
                })
                self._log_epoch()
                break

        real_next_state = next_state.copy()
        for idx, trunc in enumerate(truncated):
            if trunc:
                real_next_state[idx] = infos['final_observation'][idx]

        # Store the transition in memory
        memory.push(self._state, action, real_next_state, reward, terminated)

        # Move to the next state
        self._state = next_state

        # Perform one step of the optimization (on the policy network)
        self._sample_and_optimize(memory)

        self.global_iter += self._opts.vectorize

        if self._thumb_interval > 0 and self.global_iter % self._thumb_interval == 0 and self.global_iter > self._opts.learning_starts:
            if self._opts.render_mode == 'rgb_array_list':
                video = self._environment.envs[0].render()
                video = np.stack(video).transpose((0, 3, 1, 2))[np.newaxis, ...]
                self.writer.add_video('episodes/overviews', video, global_step=self.global_iter, fps=60)

    def _apply_soft_updates(self):
        self._soft_update(self._net['PolicyNet'], self._target_net)

    def _soft_update(self, net, target_net):
        # Soft update of the target network's weights
        # θ′ ← τθ + (1 − τ)θ′
        target_state_dict = target_net.state_dict()
        net_state_dict = net.state_dict()
        for key in net_state_dict:
            # Soft update except for batchnorm parameters.
            # https://github.com/DLR-RM/stable-baselines3/pull/1004
            if 'running_' in key:
                target_state_dict[key] = net_state_dict[key]
            else:
                target_state_dict[key] = net_state_dict[key] * self._opts.tau + target_state_dict[key] * (1 - self._opts.tau)
        target_net.load_state_dict(target_state_dict)

    def _sample_and_optimize(self, memory):
        if len(memory) < self._opts.batch_size or self.global_iter < self._opts.learning_starts:
            return

        tensors = [t.to(self._ctx[0]) for t in memory.sample(self._opts.batch_size)]
        states, actions, next_states, rewards, dones = tensors

        self._optimize(states, actions, next_states, rewards, dones)
        if self.global_iter % self._opts.target_frequency == 0:
            self._apply_soft_updates()

        self._report_stats(states)
        self._metrics_store.update_train(self._loss.decompose())

    def _optimize(self, states, actions, next_states, rewards, dones):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        with torch.cuda.amp.autocast(self._opts.fp16):
            state_action_values = self._net['PolicyNet'](states).gather(-1, actions.view(-1, 1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            estimated_rewards = self._target_net(next_states) * (1 - dones)
            next_state_action_values, _ = estimated_rewards.max(keepdims=True, dim=1)

        # Compute the expected Q values
        # r + gamma * Q(s', a)
        expected_next_state_action_values = rewards + self._opts.gamma * next_state_action_values

        # Compute temporal difference error
        loss = self._loss.compute(state_action_values, expected_next_state_action_values)

        # Same optimizer used for all models
        self._optimizer['PolicyNet'].zero_grad()
        self._scaler.scale(loss).backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._net['PolicyNet'].parameters(), 100)
        self._scaler.step(self._optimizer['PolicyNet'])
        self._scaler.update()

    def _select_action(self, state):
        if type(state) != torch.Tensor:
            state = torch.tensor(state).to(self._ctx[0])

        with torch.no_grad():
            # Pick action with the larger expected reward.
            # argmax Q(s, a)
            estimated_rewards = self._net['PolicyNet'](state)
            max_reward, max_index = estimated_rewards.max(dim=1)
            return max_index#.view(1, 1)

    def _report_stats(self, states):
        if self._stats_interval > 0 and self.global_iter % self._stats_interval == 0 and self.writer is not ...:
            self._net.eval()

            states.requires_grad = True
            estimated_rewards = self._net['PolicyNet'](states)
            jacobian_norm = calc_jacobian_norm(estimated_rewards, [states])
            gradients_norm = calc_net_gradient_norm(self._net['PolicyNet'])

            gradients_norm_hist(self.writer, 'jacobian', [jacobian_norm], self.global_iter, labels=['w.r.t. inputs'])
            gradients_norm_hist(self.writer, 'weights_grad', [gradients_norm], self.global_iter, labels=['w.r.t. weights'])

            self._net.train()

    def get_graph(self) -> torch.nn.Module:
        pass
