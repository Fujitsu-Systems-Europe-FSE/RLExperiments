from torch import nn
from time import time
from itertools import count
from model.dqn.deep_q_net import DeepQNet
from utils.env_exploration import EpsilonGreedy
from apheleia.trainer.rl_trainer import RLTrainer
from apheleia.metrics.metric_store import MetricStore
from apheleia.metrics.meters import AverageMeter, SumMeter

import torch


class DQNTrainer(RLTrainer):

    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, *args, **kwargs):
        super().__init__(opts, net, optims, scheds, loss, validator, metrics, ctx, 'RL', *args, **kwargs)
        self._add_metrics()
        self._initial_setup()

    def _initial_setup(self):
        self._env_explorer = EpsilonGreedy(self._opts, self._ctx, self._select_action)
        # TODO Move in ModelStore ?
        self._target_net = self._targetize_net(DeepQNet)

    def _targetize_net(self, netclazz):
        net = netclazz(self._opts)
        target_net = nn.DataParallel(net, device_ids=self._ctx).to(self._ctx[0])
        target_net.load_state_dict(self._net.get(net.model_name()).state_dict())
        return target_net

    def _add_metrics(self):
        self._metrics_store.add_train_metric('rewards/episode_rewards', SumMeter(''))
        self._metrics_store.add_train_metric('episodes/duration_in_steps', AverageMeter(''))
        self._metrics_store.add_train_metric('episodes/duration_in_secs', AverageMeter(''))

    def _train_loop(self, memory, *args, **kwargs):
        # Initialize the environment and get it's state
        state, info = self._environment.reset()
        state = torch.tensor(state).unsqueeze(0).to(self._ctx[0])

        t0 = time()
        for t in count():
            action = self._env_explorer.explore(state, self.global_iter)
            formatted_action = action.item() if hasattr(self._environment.action_space, 'n') else action.cpu().numpy().flatten()
            observation, reward, terminated, truncated, _ = self._environment.step(formatted_action)
            reward = torch.tensor([[reward]]).to(self._ctx[0])
            truncated = (truncated and self._opts.max_steps is None) or (t == self._opts.max_steps)
            done = terminated or truncated
            next_state = None if terminated else torch.tensor(observation).to(self._ctx[0]).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self._sample_and_optimize(memory)

            self._metrics_store.update_train({'rewards/episode_rewards': reward.item()})
            if done:
                self._metrics_store.update_train({
                    'episodes/duration_in_steps': t + 1,
                    'episodes/duration_in_secs': time() - t0
                })
                break

            self.global_iter += 1

    def _apply_soft_updates(self):
        self._soft_update(self._net['PolicyNet'], self._target_net)

    def _soft_update(self, net, target_net):
        # Soft update of the target network's weights
        # θ′ ← τθ + (1 − τ)θ′
        target_state_dict = target_net.state_dict()
        net_state_dict = net.state_dict()
        for key in net_state_dict:
            target_state_dict[key] = net_state_dict[key] * self._opts.tau + target_state_dict[key] * (1 - self._opts.tau)
        target_net.load_state_dict(target_state_dict)

    def _sample_and_optimize(self, memory):
        if len(memory) < self._opts.batch_size:
            return

        states, actions, next_states, rewards, masks = memory.sample(self._opts.batch_size)

        self._optimize(states, actions, next_states, rewards, masks)
        self._apply_soft_updates()

        # self._log_iteration(batch_idx)
        self._metrics_store.update_train(self._loss.decompose())

    def _optimize(self, states, actions, next_states, rewards, masks):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        with torch.cuda.amp.autocast(self._opts.fp16):
            state_action_values = self._net['PolicyNet'](states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            estimated_rewards = self._target_net(next_states) * masks
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
        with torch.no_grad():
            # Pick action with the larger expected reward.
            # argmax Q(s, a)
            estimated_rewards = self._net['PolicyNet'](state)
            max_reward, max_index = estimated_rewards.max(dim=1)
            return max_index.view(1, 1)

    def _save_checkpoints(self, out_filename):
        pass

    def _report_stats(self, *args):
        pass

    def get_graph(self) -> torch.nn.Module:
        pass
