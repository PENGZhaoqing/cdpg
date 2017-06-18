# print(__doc__)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold

Axes3D

import argparse
from env_2a import Env
from com_network import *
from config import Config
from utils import *

dir = os.path.dirname(__file__)
os.path.join(os.path.realpath('..'))
file_name = os.path.basename(__file__)

import numpy as np


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=100000)
    parser.add_argument('--epoch-num', type=int, default=20)
    parser.add_argument('--start-epoch', type=int, default=20)
    parser.add_argument('--testing-epoch', type=int, default=39)
    parser.add_argument('--save-log', type=str, default='log')
    parser.add_argument('--signal-num', type=int, default=4)
    parser.add_argument('--toxin', type=int, default=0)
    parser.add_argument('--a1-AC-folder', type=str, default='basic_signal_4/a1_AC')
    parser.add_argument('--a2-AC-folder', type=str, default='basic_signal_4/a2_AC')
    parser.add_argument('--a1-CDPG-folder', type=str, default='basic_signal_4/a1_CDPG')
    parser.add_argument('--a2-CDPG-folder', type=str, default='basic_signal_4/a2_CDPG')

    args = parser.parse_args()
    config = Config(args)
    q_ctx = config.ctx
    gamma = config.gamma
    np.random.seed(args.seed)
    testing_epoch = args.testing_epoch
    signal_num = args.signal_num
    toxin = args.toxin
    a1_AC_folder = args.a1_AC_folder
    a2_AC_folder = args.a2_AC_folder
    a1_CDPG_folder = args.a1_CDPG_folder
    a2_CDPG_folder = args.a2_CDPG_folder

    envs = []
    for i in range(args.num_envs):
        envs.append(Env(i, toxin))
    num_envs = len(envs)

    # envs[0].game.draw = True
    # envs[0].ple.display_screen = True

    action_set = envs[0].ple.get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)
    action_num = len(action_map1)

    # Init networks
    a1_AC = ActorCritic(state_dim=74, signal_num=signal_num, act_space=action_num, dir=dir, folder=a1_AC_folder,
                        config=config)
    a2_AC = ActorCritic(state_dim=74, signal_num=signal_num, act_space=action_num, dir=dir, folder=a2_AC_folder,
                        config=config)

    a1_CDPG = CDPG(state_dim=74, signal_num=signal_num, dir=dir, folder=a1_CDPG_folder, config=config)
    a2_CDPG = CDPG(state_dim=74, signal_num=signal_num, dir=dir, folder=a2_CDPG_folder, config=config)

    a1_AC.load_params(testing_epoch)
    a2_AC.load_params(testing_epoch)
    a1_CDPG.load_params(testing_epoch)
    a2_CDPG.load_params(testing_epoch)

    episode_step = 100

    X = []
    color = []
    count = [0, 0, 0, 0]
    while len(X) <= episode_step:

        episode_reward = np.zeros(num_envs, dtype=np.float)
        step_ob1 = []
        step_ob2 = []
        all_done = False

        for i in range(num_envs):
            envs[i].reset_var()
            envs[i].ple.reset_game()
            ob = envs[i].ple.get_states()
            step_ob1.append(ob[0])
            step_ob2.append(ob[1])

        while not all_done:

            tmp1 = step_ob1[0][0:-2].reshape(-1, 3)
            tmp2 = step_ob2[0][0:-2].reshape(-1, 3)
            if np.any(tmp1[:, 0] > 0) and np.any(tmp1[:, 1] > 0) and np.all(tmp2[:, 0] == 0):
                step_ob2[0][-2:] = 0
                signal2 = a2_CDPG.forward(step_ob1, is_train=False)[0]
                _, step_value2, _, step_policy2 = a2_AC.forward(step_ob2, signal2, is_train=False)
                step_policy2 = step_policy2.asnumpy()

                us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
                step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

                if count[step_action2[0]] <= 250:
                    count[step_action2[0]] += 1

                    x = step_policy2[0, 1] - step_policy2[0, 3]
                    y = step_policy2[0, 2] - step_policy2[0, 0]
                    angle = angle_between((0, 1), (x, y))
                    X.append(signal2.asnumpy()[0])
                    # angle /= 90.0
                    color.append(angle)

            signal1 = a1_CDPG.forward(step_ob2, is_train=False)[0]
            signal2 = a2_CDPG.forward(step_ob1, is_train=False)[0]

            _, step_value1, _, step_policy1 = a1_AC.forward(step_ob1, signal1, is_train=False)
            _, step_value2, _, step_policy2 = a2_AC.forward(step_ob2, signal2, is_train=False)

            step_policy1 = step_policy1.asnumpy()
            step_value1 = step_value1.asnumpy()
            step_policy2 = step_policy2.asnumpy()
            step_value2 = step_value2.asnumpy()

            us1 = np.random.uniform(size=step_policy1.shape[0])[:, np.newaxis]
            step_action1 = (np.cumsum(step_policy1, axis=1) > us1).argmax(axis=1)

            us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
            step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

            for i in range(num_envs):
                if not envs[i].done:
                    next_ob, reward, done = envs[i].ple.act(
                        [action_map1[step_action1[i]], action_map2[step_action2[i]]])

                    envs[i].add(step_ob1[i], step_ob2[i], step_action1[i], step_action2[i],
                                step_value1[i][0],
                                step_value2[i][0],
                                reward[0], reward[1])

                    envs[i].done = done
                    step_ob1[i] = next_ob[0]
                    step_ob2[i] = next_ob[1]
                    episode_reward[i] += sum(reward)

            for i in range(num_envs):
                envs[i].clear()
            all_done = np.all([envs[i].done for i in range(num_envs)])

        print len(X)
        print str(episode_reward)

    n_components = 2

    for num in np.arange(10, 40, 10):
        fig = plt.figure(figsize=(15, 8))
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=num)
        Y = tsne.fit_transform(X)
        # ax = fig.add_subplot(1, 1, 1)
        sc = plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.hsv)
        sc.axes.get_xaxis().set_visible(False)
        sc.axes.get_yaxis().set_visible(False)
        # sc.xaxis.set_major_formatter(NullFormatter())
        # sc.yaxis.set_major_formatter(NullFormatter())
        clb = plt.colorbar(sc)
        clb.ax.set_title(r'Angle $\theta$ (degree)')

        # tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=num)
        # Y = tsne.fit_transform(X)
        # ax = fig.add_subplot(1, 2, 2)
        # sc = plt.scatter(Y[:, 0], Y[:, 1], s=20, c=color, cmap=plt.cm.viridis)
        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        # plt.colorbar(sc)

        plt.axis('tight')
        plt.show()


if __name__ == '__main__':
    main()
