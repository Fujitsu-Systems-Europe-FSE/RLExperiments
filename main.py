from apheleia.app import App

from dataset.replay_memory import ReplayMemory
from trainer.dqn_trainer import DQNTrainer
from model.losses.huber_loss import HuberLoss
from apheleia.utils.initialization import seed
from apheleia.utils.logger import ProjectLogger
from model.dqn.deep_q_net import TargetNet, PolicyNet
from apheleia.metrics.metric_store import MetricStore
from apheleia.catalog import PipelinesCatalog, LossesCatalog
from apheleia.factory.training_pipeline_factory import TrainingPipelineFactory

import matplotlib
import gymnasium as gym
import matplotlib.pyplot as plt

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


def train(args, ctx):
    seed(args)

    state, info = env.reset()
    env.render()
    args.env = env
    # Get the number of state observations
    args.n_observations = len(state)
    # Get number of actions from gym action space
    args.n_actions = env.action_space.n
    args.hidden_dim = 128

    if args.runs > 1:
        root_path = args.outdir
        # basename = os.path.basename(root_path)
        # for i in range(1, args.runs + 1):
        #     args.outdir = os.path.join(root_path, f'{basename}-{i}')
        #     init_train(args, ctx, train_data, val_data, test_data)
    else:
        init_train(args, ctx)


def init_train(opts, ctx, *args):
    trainer = TrainingPipelineFactory(opts, ctx).build()

    memory = ReplayMemory(10000)

    try:
        trainer.start(memory, None, None)
    except KeyboardInterrupt as interrupt:
        ProjectLogger().warning('Keyboard interrupt received. Checkpointing current epoch.')
        trainer.do_interrupt_backup()
        raise interrupt


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    rl_exp = App('RL-Experiments', with_dataset=False)
    rl_exp.add_bootstrap('train', train)
    rl_exp.run()

