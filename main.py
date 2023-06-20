from apheleia.app import App
from model.ddpg.actor import Actor
from model.ddpg.critic import Critic
from model.dqn.deep_q_net import DeepQNet
from model.losses.ddpg_loss import DDPGLoss
from trainer.dqn_trainer import DQNTrainer
from model.losses.huber_loss import HuberLoss
from dataset.replay_memory import ReplayMemory
from apheleia.metrics.metric_store import MetricStore
from trainer.actor_critic_trainer import ActorCriticTrainer
from apheleia.catalog import PipelinesCatalog, LossesCatalog

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
    'DQN': {
        'models': [DeepQNet],
        'optimizers': ['adamw'],
        'losses': ['huber'],
        'trainer': DQNTrainer,
        'metrics': MetricStore
    },
    'DDPG': {
        'models': [Actor, Critic],
        'optimizers': ['adam', 'adam'],
        'losses': ['ddpg'],
        'trainer': ActorCriticTrainer,
        'metrics': MetricStore
    }
}


def setup_train_env(args):
    env = gym.make(args.env_name, render_mode=args.render_mode)
    state, info = env.reset()

    args.env = env
    # Get the number of state observations
    args.n_states = len(state)
    # Env is discrete
    if hasattr(env.action_space, 'n'):
        args.n_actions = int(env.action_space.n)
    else:
        args.n_actions = env.action_space.shape[0]
        args.min_action = float(env.action_space.low[0])
        args.max_action = float(env.action_space.high[0])

    return ReplayMemory(int(args.mem_size)), None, None


if __name__ == '__main__':
    # from vizdoom import gymnasium_wrapper
    # env = gym.make("VizdoomHealthGatheringSupreme-v0", render_mode="human")

    # # Rendering random rollouts for ten episodes
    # for _ in range(10):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    #         done = terminated or truncated

    rl_exp = App('RL-Experiments', with_dataset=False)
    train_parser = rl_exp.cli.get_subparser('train')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Networks hidden dimension')
    train_parser.add_argument('--max-steps', type=int, help='Max steps during an episode')
    train_parser.add_argument('--mem-size', type=int, default=1e6, help='Replay memory size')
    train_parser.add_argument('--gamma', type=int, default=0.99, help='Discount value')
    train_parser.add_argument('--tau', type=int, default=5e-3, help='Soft update merging factor')
    train_parser.add_argument('--env', dest='env_name', type=str, default='MountainCar-v0', help='Training environment name')
    train_parser.add_argument('--render-mode', type=str, default=None, help='Environment render mode')

    rl_exp.add_bootstrap('train', setup_train_env)
    rl_exp.run()

