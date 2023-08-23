from trainer.dqn_trainer import DQNTrainer
from apheleia.metrics.metric_store import MetricStore
from utils.env_exploration import Gaussian, OrnsteinUlhenbeck
from apheleia.utils.visualize import gradients_norm_hist
from apheleia.utils.gradients import calc_jacobian_norm, calc_net_gradient_norm

import torch


class ActorCriticTrainer(DQNTrainer):
    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, *args, **kwargs):
        super().__init__(opts, net, optims, scheds, loss, validator, metrics, ctx, 'DDPG', *args, **kwargs)

    def _initial_setup(self):
        # self._env_explorer = Gaussian(self._opts, self._ctx, self._select_action)
        self._env_explorer = OrnsteinUlhenbeck(self._opts, self._ctx, self._select_action)
        # TODO Move in ModelStore ?

        actor_clazz = self._net['Actor'].module.__class__
        critic_clazz = self._net['Critic'].module.__class__
        self._target_actor = self._targetize_net(actor_clazz)
        self._target_critic = self._targetize_net(critic_clazz)

    def _apply_soft_updates(self):
        self._soft_update(self._net['Actor'], self._target_actor)
        self._soft_update(self._net['Critic'], self._target_critic)

    def _optimize_critic(self, states, actions, next_states, rewards, dones):
        # compute state_action_values for St+1
        with torch.no_grad():
            # produce a continuous action given a state
            estimated_actions = self._target_actor(next_states)
            # evaluate actions quality for a given state
            next_state_action_values = self._target_critic(next_states, estimated_actions) * (1 - dones)

            expected_next_state_action_values = rewards + self._opts.gamma * next_state_action_values

        # state_action_values for St
        with torch.cuda.amp.autocast(self._opts.fp16):
            # evaluate actions quality for a given state
            state_action_values = self._net['Critic'](states, actions)
            critic_loss = self._loss.compute_critic_loss(state_action_values, expected_next_state_action_values)

        # Optimize critic
        self._optimizer['Critic'].zero_grad()
        self._scaler.scale(critic_loss).backward()
        self._scaler.step(self._optimizer['Critic'])
        self._scaler.update()

    def _optimize_actor(self, states):
        with torch.cuda.amp.autocast(self._opts.fp16):
            actions = self._net['Actor'](states)
            actor_loss = self._loss.compute_actor_loss(self._net['Critic'], actions, states)

        # Optimize the actor
        self._optimizer['Actor'].zero_grad()
        self._scaler.scale(actor_loss).backward()
        self._scaler.step(self._optimizer['Actor'])
        self._scaler.update()

    def _optimize(self, states, actions, next_states, rewards, masks):
        self._optimize_critic(states, actions, next_states, rewards, masks)
        if self.global_iter % self._opts.policy_frequency == 0:
            self._optimize_actor(states)

    def _select_action(self, state):
        if type(state) != torch.Tensor:
            state = torch.tensor(state).to(self._ctx[0])

        with torch.no_grad():
            action = self._net['Actor'](state)

        return action

    def _report_stats(self, states):
        if self._stats_interval > 0 and self.global_iter % self._stats_interval == 0 and self.writer is not ...:
            self._net.eval()

            states.requires_grad = True
            actions = self._net['Actor'](states)
            actor_jacobian_norm = calc_jacobian_norm(actions, [states])
            actor_gradients_norm = calc_net_gradient_norm(self._net['Actor'])

            state_action_values = self._net['Critic'](states, actions)
            critic_jacobian_norm = calc_jacobian_norm(state_action_values, [actions])
            critic_gradients_norm = calc_net_gradient_norm(self._net['Critic'])

            gradients_norm_hist(self.writer, 'jacobian', [actor_jacobian_norm, critic_jacobian_norm], self.global_iter, labels=['w.r.t. inputs (actor)', 'w.r.t. inputs (critic)'])
            gradients_norm_hist(self.writer, 'weights_grad', [actor_gradients_norm, critic_gradients_norm], self.global_iter, labels=['w.r.t. weights (actor)', 'w.r.t. weights (critic)'])

            self._net.train()

    def get_graph(self) -> torch.nn.Module:
        pass
