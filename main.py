from apheleia.app import App
from trainer.dqn_trainer import DQNTrainer
from model.losses.huber_loss import HuberLoss
from dataset.replay_memory import ReplayMemory
from model.dqn.deep_q_net import TargetNet, PolicyNet
from apheleia.metrics.metric_store import MetricStore
from apheleia.catalog import PipelinesCatalog, LossesCatalog

import gymnasium as gym

LossesCatalog()['RL'] = {
    'huber': {
        'class': HuberLoss
    }
}
PipelinesCatalog()['RL'] = {
    'DQN': {
        'models': [TargetNet, PolicyNet],
        'optimizers': ['adamw'],
        'losses': ['huber'],
        'trainer': DQNTrainer,
        'metrics': MetricStore
    }
}


def setup_train_env(args):
    env = gym.make('CartPole-v1', render_mode='human')
    state, info = env.reset()
    env.render()
    args.env = env
    # Get the number of state observations
    args.n_observations = len(state)
    # Get number of actions from gym action space
    args.n_actions = env.action_space.n
    args.hidden_dim = 128
    return ReplayMemory(10000), None, None


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
    rl_exp.add_bootstrap('train', setup_train_env)
    rl_exp.run()

