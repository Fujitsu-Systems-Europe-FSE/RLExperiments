from model.ddpg.actor import Actor
from model.ddpg.critic import Critic
from trainer.dqn_trainer import DQNTrainer
from utils.env_exploration import Gaussian
from apheleia.metrics.metric_store import MetricStore

import torch


class ActorCriticTrainer(DQNTrainer):
    def __init__(self, opts, net, optims, scheds, loss, validator, metrics: MetricStore, ctx, *args, **kwargs):
        super().__init__(opts, net, optims, scheds, loss, validator, metrics, ctx, 'DDPG', *args, **kwargs)

    def _initial_setup(self):
        self._env_explorer = Gaussian(self._opts, self._ctx, self._select_action)
        # TODO Move in ModelStore ?
        self._target_actor = self._targetize_net(Actor)
        self._target_critic = self._targetize_net(Critic)

    def _apply_soft_updates(self):
        self._soft_update(self._net['Actor'], self._target_actor)
        self._soft_update(self._net['Critic'], self._target_critic)

    def _optimize(self, states, actions, next_states, rewards, masks):
        # compute state_action_values for St+1
        with torch.no_grad():
            # produce a continuous action given a state
            estimated_actions = self._target_actor(next_states)
            # evaluate actions quality for a given state
            next_state_action_values = self._target_critic(next_states, estimated_actions) * masks

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

        with torch.cuda.amp.autocast(self._opts.fp16):
            actor_loss = self._loss.compute_actor_loss(self._net['Critic'], self._net['Actor'], states)

        # Optimize the actor
        self._optimizer['Actor'].zero_grad()
        self._scaler.scale(actor_loss)
        self._scaler.step(self._optimizer['Actor'])
        self._scaler.update()

    def _select_action(self, state):
        self._net['Actor'].eval()
        with torch.no_grad():
            action = self._net['Actor'](state)
        self._net['Actor'].train()
        return action

    def _save_checkpoints(self, out_filename):
        pass

    def _report_stats(self, *args):
        pass

    def get_graph(self) -> torch.nn.Module:
        pass
