import argparse
import logging
from env_2a import Env
from uncom_network import *
from config import Config
from utils import *

dir = os.path.dirname(__file__)
os.path.join(os.path.realpath('..'))
file_name = os.path.basename(__file__)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--testing', type=int, default=1)
    parser.add_argument('--epoch-num', type=int, default=20)
    parser.add_argument('--start-epoch', type=int, default=39)
    parser.add_argument('--testing-epoch', type=int, default=30)
    parser.add_argument('--signal-num', type=int, default=4)
    parser.add_argument('--toxin', type=int, default=0)
    parser.add_argument('--a1-AC-folder', type=str, default='basic/a1_AC')
    parser.add_argument('--a2-AC-folder', type=str, default='basic/a2_AC')

    args = parser.parse_args()
    config = Config(args)
    q_ctx = config.ctx
    gamma = config.gamma
    np.random.seed(args.seed)
    testing_epoch = args.testing_epoch
    signal_num = args.signal_num
    a1_AC_folder = args.a1_AC_folder
    a2_AC_folder = args.a2_AC_folder

    a1_AC = ActorCritic(state_dim=74, act_space=4, dir=dir, folder=a1_AC_folder,
                        config=config)
    a2_AC = ActorCritic(state_dim=74, act_space=4, dir=dir, folder=a2_AC_folder,
                        config=config)

    a1_AC.load_params(testing_epoch)
    a2_AC.load_params(testing_epoch)

    print_params(logging, a1_AC.model)

    X = np.arange(0, 1, 0.02)
    Y = np.arange(0, 1, 0.02)
    Z = np.zeros((X.shape[0], Y.shape[0], 4))

    for j in range(X.shape[0]):
        step_ob1 = np.zeros((74,))
        step_ob2 = np.zeros((74,))
        step_ob1[54] = X[j]

        for k in range(Y.shape[0]):
            step_ob1[19] = X[k]
            step_ob2[55] = Y[k]

            data_batch1 = mx.io.DataBatch(data=[mx.nd.array([step_ob1], ctx=q_ctx)], label=None)
            a1_AC.model.reshape([('data', (1, 74))])
            a1_AC.model.forward(data_batch1, is_train=False)
            _, step_value1, _, step_policy1 = a1_AC.model.get_outputs()

            data_batch2 = mx.io.DataBatch(data=[mx.nd.array([step_ob2], ctx=q_ctx)], label=None)
            a2_AC.model.reshape([('data', (1, 74))])
            a2_AC.model.forward(data_batch2, is_train=False)
            _, step_value2, _, step_policy2 = a2_AC.model.get_outputs()

            step_policy1 = step_policy1.asnumpy()
            step_policy2 = step_policy2.asnumpy()

            Z[j, k] = step_policy2[0]

    Z = np.array(Z)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z[:, :, 0], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Fellow distance')
    ax.set_ylabel('Prey distance')
    ax.set_zlabel('prob')
    ax.set_title("Down Action")

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, Z[:, :, 1], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Fellow distance')
    ax.set_ylabel('Prey distance')
    ax.set_zlabel('prob')
    ax.set_title("Right Action")

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(X, Y, Z[:, :, 2], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Fellow distance')
    ax.set_ylabel('Prey distance')
    ax.set_zlabel('prob')
    ax.set_title("Up Action")

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax.plot_surface(X, Y, Z[:, :, 3], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Fellow distance')
    ax.set_ylabel('Prey distance')
    ax.set_zlabel('prob')
    ax.set_title("Left Action")

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == '__main__':
    main()
