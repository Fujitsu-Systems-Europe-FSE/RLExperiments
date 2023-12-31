from torchvision import transforms
from functools import partial

from apheleia import ProjectLogger
from apheleia.app import App
from vizdoom import gymnasium_wrapper
from model.ddpg.actor_cnn import Actor as CnnActor
from model.ddpg.critic_cnn import Critic as CnnCritic
from model.ddpg.actor_dense import Actor as DenseActor
from model.ddpg.critic_dense import Critic as DenseCritic
from model.dqn.deep_q_net_dense import DeepQNet as DenseDQN
from model.dqn.deep_q_net_cnn import DeepQNet as CnnDQN
from model.losses.ddpg_loss import DDPGLoss
from trainer.dqn_trainer import DQNTrainer
from model.losses.huber_loss import HuberLoss
from dataset.replay_memory import ReplayMemory
from apheleia.metrics.metric_store import MetricStore
from trainer.actor_critic_trainer import ActorCriticTrainer
from apheleia.catalog import PipelinesCatalog, LossesCatalog

import numpy as np
import gymnasium as gym

LossesCatalog()['RL'] = {
    'huber': {
        'class': HuberLoss
    },
    'ddpg': {
        'class': DDPGLoss
    }
}
PipelinesCatalog()['RL'] = {
    'dense-dqn': {
        'models': [DenseDQN],
        'optimizers': ['adamw'],
        'losses': ['huber'],
        'trainer': DQNTrainer,
        'metrics': MetricStore
    },
    'cnn-dqn': {
        'models': [CnnDQN],
        'optimizers': ['adamw'],
        'losses': ['huber'],
        'trainer': DQNTrainer,
        'metrics': MetricStore
    },
    'dense-ddpg': {
        'models': [DenseActor, DenseCritic],
        'optimizers': ['adam', 'adam'],
        'losses': ['ddpg'],
        'trainer': ActorCriticTrainer,
        'metrics': MetricStore
    },
    'cnn-ddpg': {
        'models': [CnnActor, CnnCritic],
        'optimizers': ['adam', 'adam'],
        'losses': ['ddpg'],
        'trainer': ActorCriticTrainer,
        'metrics': MetricStore
    }
}


def make_env(num, args):
    render_mode = args.render_mode if num == 0 else 'rgb_array_list'
    env = gym.make(args.env_name, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


def setup_train_env(args):
    memory = ReplayMemory(int(args.mem_size))
    if args.env_name.startswith('Vizdoom'):
        memory.add_transforms([
            transforms.Lambda(lambda x: x['screen']),
            transforms.Lambda(lambda x: np.stack([transforms.ToTensor()(t.squeeze()) for t in np.split(x, x.shape[0], axis=0)], axis=0))
        ])

    env = gym.vector.SyncVectorEnv([partial(make_env, i, args) for i in range(args.vectorize)])

    state, info = env.reset()

    args.env = env
    # Get the number of state observations
    args.n_states = state.shape[-1] if type(state) == np.ndarray else 512
    if type(env.single_action_space) == gym.spaces.Box:
        action_space = env.single_action_space.spaces['continuous'] if hasattr(env.single_action_space, 'spaces') else env.single_action_space
        args.n_actions = action_space.shape[0]
        args.min_actions = action_space.low
        args.max_actions = action_space.high
    else:
        # Env is discrete
        args.n_actions = int(env.single_action_space.n)

    return memory, None, None


if __name__ == '__main__':
    rl_exp = App('RL-Experiments', with_dataset=False)
    train_parser = rl_exp.cli.get_subparser('train')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Networks hidden dimension')
    train_parser.add_argument('--max-steps', type=int, help='Max steps during an episode')
    train_parser.add_argument('--mem-size', type=int, default=int(1e6), help='Replay memory size')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount value')
    train_parser.add_argument('--tau', type=float, default=5e-3, help='Soft update merging factor')
    train_parser.add_argument('--learning-starts', type=int, default=25e3, help='Start learning after trying the given number of actions')
    train_parser.add_argument('--target-frequency', type=int, default=1, help='Frequency of training target networks in steps (delayed)')
    train_parser.add_argument('--policy-frequency', type=int, default=1, help='Frequency of training policy in steps  (delayed)')
    train_parser.add_argument('--env', dest='env_name', type=str, default='MountainCar-v0', help='Training environment name')
    train_parser.add_argument('--render-mode', type=str, choices=['human', 'rgb_array_list'], default='rgb_array_list', help='Environment render mode')
    # TODO Batchnorm currently has detrimental effects
    train_parser.add_argument('--with-bn', action='store_true', help='Enables BatchNorm layers in architectures')
    train_parser.add_argument('--vectorize', type=int, default=1, help='Number of vectorized environments for parallel env exploration')

    rl_exp.add_bootstrap('train', setup_train_env)
    rl_exp.run()

