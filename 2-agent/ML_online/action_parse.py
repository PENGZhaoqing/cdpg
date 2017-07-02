import argparse
import logging
from env_2a import Env
from com_network import *
from uncom_network import ActorCritic_uncom
from config import Config
from utils import *

dir = os.path.dirname(__file__)
os.path.join(os.path.realpath('..'))
file_name = os.path.basename(__file__)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.image as mpimg

Axes3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--testing', type=int, default=1)
    parser.add_argument('--epoch-num', type=int, default=20)
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
    testing_epoch_list = [1, 15, 30]
    signal_num = args.signal_num
    a1_AC_folder = args.a1_AC_folder
    a2_AC_folder = args.a2_AC_folder
    a1_CDPG_folder = args.a1_CDPG_folder
    a2_CDPG_folder = args.a2_CDPG_folder

    fig = plt.figure(figsize=(20, 8))
    # ax = fig.add_subplot(3, 1, 1)
    # img = mpimg.imread('loca.png')
    # plt.axis('off')
    # plt.imshow(img,interpolation='none')

    count = 1

    for testing_epoch in testing_epoch_list:

        a1_AC = ActorCritic_uncom(state_dim=74, act_space=4, dir=dir, folder='../IL_online/basic/a1_AC',
                                  config=config)
        a2_AC = ActorCritic_uncom(state_dim=74, act_space=4, dir=dir, folder='../IL_online/basic/a2_AC',
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

        ax = fig.add_subplot(2, 3, count, projection='3d')
        surf = ax.plot_surface(X, Y, Z[:, :, 3], cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('Agent1 dist.')
        ax.set_ylabel('Prey dist.')
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$\pi$(left)', labelpad=10)
        ax.set_zlim(0, 1)
        ax.set_title("IL (epoch " + str(testing_epoch) + ")", fontsize=12)
        count += 1


    for testing_epoch in testing_epoch_list:

        a1_AC = ActorCritic(state_dim=74, signal_num=signal_num, act_space=4, dir=dir, folder=a1_AC_folder,
                            config=config)
        a2_AC = ActorCritic(state_dim=74, signal_num=signal_num, act_space=4, dir=dir, folder=a2_AC_folder,
                            config=config)

        a1_CDPG = CDPG(state_dim=74, signal_num=signal_num, dir=dir, folder=a1_CDPG_folder, config=config)
        a2_CDPG = CDPG(state_dim=74, signal_num=signal_num, dir=dir, folder=a2_CDPG_folder, config=config)

        a1_AC.load_params(testing_epoch)
        a2_AC.load_params(testing_epoch)
        a1_CDPG.load_params(testing_epoch)
        a2_CDPG.load_params(testing_epoch)

        print_params(logging, a1_AC.model)
        print_params(logging, a1_CDPG.model)

        X = np.arange(0, 1, 0.02)
        Y = np.arange(0, 1, 0.02)
        Z = np.zeros((X.shape[0], Y.shape[0], 4))

        for j in range(X.shape[0]):
            step_ob1 = np.zeros((74,))
            step_ob2 = np.zeros((74,))
            step_ob1[54] = X[j]

            for k in range(Y.shape[0]):
                step_ob1[19] = X[k]
                step_ob2[55] = X[k]

                signal1 = a1_CDPG.forward([step_ob2], is_train=False)[0]
                signal2 = a2_CDPG.forward([step_ob1], is_train=False)[0]

                _, step_value1, _, step_policy1 = a1_AC.forward([step_ob1], signal1, is_train=False)
                _, step_value2, _, step_policy2 = a2_AC.forward([step_ob2], signal2,
                                                                is_train=False)
                step_policy1 = step_policy1.asnumpy()
                step_policy2 = step_policy2.asnumpy()

                Z[j, k] = step_policy2[0]

        Z = np.array(Z)
        X, Y = np.meshgrid(X, Y)

        ax = fig.add_subplot(2, 3, count, projection='3d')
        surf = ax.plot_surface(X, Y, Z[:, :, 3], cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel('Agent1 dist.')
        ax.set_ylabel('Prey dist.')
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$\pi$(left)', labelpad=10)
        ax.set_zlim(0, 1)

        if testing_epoch == 30:
            cbaxes = fig.add_axes([0.95, 0.2, 0.01, 0.18])
            cb = fig.colorbar(surf, cax=cbaxes)
            # fig.colorbar(surf, shrink=0.4, aspect=10, orientation="vertical", pad=0.12)
        ax.set_title("CML (epoch " + str(testing_epoch) + ")", fontsize=12)

        count += 1

        # Customize the z axis.
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # plt.tight_layout(pad=10, w_pad=10, h_pad=10)
    plt.show()


if __name__ == '__main__':
    main()

    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # surf = ax.plot_surface(X, Y, Z[:, :, 0], cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # ax.set_xlabel('Fellow distance')
    # ax.set_ylabel('Prey distance')
    # ax.set_zlabel('prob')
    # ax.set_title("Down Action")
    #
    # ax = fig.add_subplot(2, 2, 2, projection='3d')
    # surf = ax.plot_surface(X, Y, Z[:, :, 1], cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # ax.set_xlabel('Fellow distance')
    # ax.set_ylabel('Prey distance')
    # ax.set_zlabel('prob')
    # ax.set_title("Right Action")
    #
    # ax = fig.add_subplot(2, 2, 3, projection='3d')
    # surf = ax.plot_surface(X, Y, Z[:, :, 2], cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # ax.set_xlabel('Fellow distance')
    # ax.set_ylabel('Prey distance')
    # ax.set_zlabel('prob')
    # ax.set_title("Up Action")
