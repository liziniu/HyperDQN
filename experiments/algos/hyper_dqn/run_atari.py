import os
import torch
import argparse
import numpy as np
from pprint import pprint

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector

from experiments.algos.hyper_dqn.buffer import VectorReplayBuffer
from experiments.algos.hyper_dqn.policy import HyperDQNPolicy
from experiments.algos.hyper_dqn.network.atari_network import HyperDQN
from experiments.algos.hyper_dqn.utils import LinearSchedule
from experiments.utils.offpolicy import offpolicy_trainer
from experiments.utils.logger import Logger
from experiments.utils.csv_writer import CSVWriter
from experiments.utils.io_utils import creat_log_dir, save_config, save_code
from experiments.envs.atari_wrapper import wrap_atari


def get_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=2021)
    # algorithm
    parser.add_argument('--agent', type=str, default='hyper_dqn')
    parser.add_argument('--prior-scale', type=float, default=0.0)
    parser.add_argument('--posterior-scale', type=float, default=1.0)
    parser.add_argument('--prior-mean', type=float, default=0.0)
    parser.add_argument('--prior-std', type=float, default=1.0)
    parser.add_argument('--num-train-iter', type=int, default=10)
    parser.add_argument('--z-size', type=int, default=32)
    parser.add_argument('--noise-scale', type=float, default=1e-2)
    parser.add_argument('--bias-coef', type=float, default=1e-2)
    parser.add_argument('--l2-norm', type=float, default=1e-2)
    # eps-greedy
    parser.add_argument('--eps-greedy', type=int, default=0, choices=[0, 1])
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.1)
    parser.add_argument('--eps-total-step', type=int, default=int(1e6))
    # interaction
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--learning-start', type=int, default=int(5e4))
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=int(1e4))
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=25000)
    parser.add_argument('--step-per-collect', type=int, default=4)
    parser.add_argument('--update-per-step', type=float, default=0.25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=None)
    return parser.parse_args()


def make_atari_env(args):
    return wrap_atari(args.env, frame_stack=args.frames_stack, episode_life=True)


def make_atari_env_watch(args):
    return wrap_atari(args.env, frame_stack=args.frames_stack, episode_life=False, clip_rewards=False,
                      max_episode_steps=int(27e3))


def test_hyper_dqn(args=get_args()):
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # make environments
    train_envs = SubprocVectorEnv([lambda: make_atari_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # define model
    net = HyperDQN(
        *args.state_shape, args.action_shape,
        z_size=args.z_size, bias_coef=args.bias_coef,
        prior_scale=args.prior_scale, posterior_scale=args.posterior_scale,
        prior_mean=args.prior_mean, prior_std=args.prior_scale,
        device=args.device
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    print("HyperDQN params:")
    param_dict = {"Non-trainable": [], "Trainable": []}
    for name, param in net.named_parameters():
        if not param.requires_grad:
            param_dict["Non-trainable"].append(name)
        else:
            param_dict["Trainable"].append(name)
    pprint(param_dict)

    policy = HyperDQNPolicy(net, optim, args.gamma, args.n_step,
                            target_update_freq=args.target_update_freq,
                            noise_scale=args.noise_scale,
                            l2_norm=args.l2_norm,
                            num_train_iter=args.num_train_iter,)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        size=args.buffer_size, buffer_num=len(train_envs),
        mask_prob=0.0, noise_dim=args.z_size, num_ensemble=1,
        ignore_obs_next=True,
        save_only_last_obs=True, stack_num=args.frames_stack)

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    args.task = args.env
    log_path = creat_log_dir(args)
    save_config(args.__dict__, log_path)
    save_code(log_path)
    writer = CSVWriter(log_path)
    logger = Logger(log_path, writer)

    # should be N_FRAMES x H x W
    logger.info("Observations shape: {}".format(args.state_shape))
    logger.info("Actions shape: {}".format(args.action_shape))

    # eps_schedule
    epsilon_schedule = LinearSchedule(
        begin_value=args.eps_train, end_value=args.eps_train_final,
        begin_t=args.learning_start, end_t=args.eps_total_step,
    )

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_fn(epoch, env_step):
        if args.eps_greedy:
            eps = epsilon_schedule(env_step)
        else:
            eps = 1.0 if env_step <= args.learning_start else 0.0
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        if args.eps_greedy:
            eps = args.eps_test
        else:
            eps = 1.0 if env_step <= args.learning_start else 0.0
        policy.set_eps(eps)

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, 5,
            args.batch_size, train_fn=train_fn, test_fn=test_fn,
            save_fn=save_fn, logger=logger,
            update_per_step=args.update_per_step, learning_start=args.learning_start,
            test_in_train=False)

        for key, val in result.items():
            logger.info("{}:{}".format(key, val))

    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=10 if args.resume_path else 10, render=args.render)
    logger.info(f'Final reward mean: {result["rews"].mean()}, reward max: {result["rews"].max()}, '
                f'length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_hyper_dqn(get_args())
