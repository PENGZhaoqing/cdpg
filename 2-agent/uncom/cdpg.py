from uncom_network import *
import argparse
import logging
from utils import *
import numpy as np
import mxnet as mx
from config import Config
from env_2a import Env
import time

dir = os.path.dirname(__file__)
os.path.join(os.path.realpath('..'))
file_name = os.path.basename(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--t-max', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=100000)
    parser.add_argument('--testing', type=int, default=0)
    parser.add_argument('--continue-training', type=int, default=0)
    parser.add_argument('--epoch-num', type=int, default=20)
    parser.add_argument('--start-epoch', type=int, default=20)
    parser.add_argument('--testing-epoch', type=int, default=19)
    parser.add_argument('--save-log', type=str, default='log')
    parser.add_argument('--signal-num', type=int, default=4)
    parser.add_argument('--toxin', type=int, default=0)
    parser.add_argument('--a1-AC-folder', type=str, default='a1_AC')
    parser.add_argument('--a2-AC-folder', type=str, default='a2_AC')

    args = parser.parse_args()
    config = Config(args)
    t_max = args.t_max
    q_ctx = config.ctx
    gamma = config.gamma
    steps_per_epoch = args.steps_per_epoch
    np.random.seed(args.seed)
    start_epoch = args.start_epoch
    testing_epoch = args.testing_epoch
    save_log = args.save_log
    epoch_num = args.epoch_num
    epoch_range = range(epoch_num)
    signal_num = args.signal_num
    toxin = args.toxin

    a1_AC_folder = args.a1_AC_folder
    a2_AC_folder = args.a2_AC_folder

    testing = args.testing
    testing = True if testing == 1 else False
    continue_training = args.continue_training
    continue_training = True if continue_training == 1 else False

    # Init envs and actions
    envs = []
    for i in range(args.num_envs):
        envs.append(Env(i, toxin))
    num_envs = len(envs)

    action_set = envs[0].ple.get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)
    action_num = len(action_map1)

    # Init networks
    a1_AC = ActorCritic(state_dim=74, act_space=action_num, dir=dir, folder=a1_AC_folder,
                        config=config)
    a2_AC = ActorCritic(state_dim=74, act_space=action_num, dir=dir, folder=a2_AC_folder,
                        config=config)

    if testing:
        envs[0].ple.force_fps = False
        envs[0].game.draw = True
        envs[0].ple.display_screen = True
        a1_AC.model.load_params(testing_epoch)
        a2_AC.model.load_params(testing_epoch)
    elif continue_training:
        epoch_range = range(start_epoch, epoch_num + start_epoch)
        a1_AC.load_params(start_epoch - 1)
        a2_AC.load_params(start_epoch - 1)
        logging_config(logging, dir, save_log, 'uncom_t_max'+str(t_max))
    else:
        logging_config(logging, dir, save_log, 'uncom_t_max'+str(t_max))

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(logging, a1_AC.model)

    for epoch in epoch_range:
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()

        while steps_left > 0:
            episode += 1
            time_episode_start = time.time()
            collisions = [0.0] * len(envs)
            episode_reward = np.zeros(num_envs, dtype=np.float)
            episode_step = 0

            step_ob1 = []
            step_ob2 = []
            for i in range(num_envs):
                envs[i].reset_var()
                envs[i].ple.reset_game()
                ob = envs[i].ple.get_states()
                step_ob1.append(ob[0])
                step_ob2.append(ob[1])

            all_done = False
            t = 1
            training_steps = 0.0
            episode_advs1 = 0
            episode_advs2 = 0
            episode_values1 = 0
            episode_values2 = 0
            episode_policys1 = 0
            episode_policys2 = 0

            while not all_done:

                data_batch1 = mx.io.DataBatch(data=[mx.nd.array(step_ob1, ctx=q_ctx)], label=None)
                a1_AC.model.reshape([('data', (num_envs, 74))])
                a1_AC.model.forward(data_batch1, is_train=False)
                _, step_value1, _, step_policy1 = a1_AC.model.get_outputs()

                data_batch2 = mx.io.DataBatch(data=[mx.nd.array(step_ob2, ctx=q_ctx)], label=None)
                a2_AC.model.reshape([('data', (num_envs, 74))])
                a2_AC.model.forward(data_batch2, is_train=False)
                _, step_value2, _, step_policy2 = a2_AC.model.get_outputs()

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
                        if reward[0] < 0:
                            collisions[i] += 1
                        episode_step += 1

                if t == t_max:

                    data_batch1 = mx.io.DataBatch(data=[mx.nd.array(step_ob1, ctx=q_ctx)], label=None)
                    a1_AC.model.forward(data_batch1, is_train=False)
                    _, extra_value1, _, _ = a1_AC.model.get_outputs()

                    data_batch2 = mx.io.DataBatch(data=[mx.nd.array(step_ob2, ctx=q_ctx)], label=None)
                    a2_AC.model.forward(data_batch2, is_train=False)
                    _, extra_value2, _, _ = a2_AC.model.get_outputs()

                    extra_value1 = extra_value1.asnumpy()
                    extra_value2 = extra_value2.asnumpy()

                    for i in range(num_envs):
                        if envs[i].done:
                            envs[i].value1.append(0)
                            envs[i].value2.append(0)
                        else:
                            envs[i].value1.append(extra_value1[i][0])
                            envs[i].value2.append(extra_value2[i][0])

                    a1_advs = []
                    a2_advs = []
                    for i in range(num_envs):
                        R1 = envs[i].value1[-1]
                        R2 = envs[i].value2[-1]
                        tmp_R1 = []
                        tmp_R2 = []
                        for j in range(len(envs[i].reward1))[::-1]:
                            R1 = envs[i].reward1[j] + gamma * R1
                            R2 = envs[i].reward2[j] + gamma * R2
                            tmp_R1.append(R1 - envs[i].value1[j])
                            tmp_R2.append(R2 - envs[i].value2[j])
                        a1_advs.extend(tmp_R1[::-1])
                        a2_advs.extend(tmp_R2[::-1])

                    a1_neg_advs_v = -np.asarray(a1_advs)
                    a2_neg_advs_v = -np.asarray(a2_advs)

                    env_action1 = []
                    for i in range(num_envs):
                        env_action1.extend(envs[i].action1)
                    neg_advs_np1 = np.zeros((len(a1_advs), action_num), dtype=np.float32)
                    neg_advs_np1[np.arange(neg_advs_np1.shape[0]), env_action1] = a1_neg_advs_v
                    neg_advs1 = mx.nd.array(neg_advs_np1, ctx=q_ctx)

                    env_action2 = []
                    for i in range(num_envs):
                        env_action2.extend(envs[i].action2)
                    neg_advs_np2 = np.zeros((len(a2_advs), action_num), dtype=np.float32)
                    neg_advs_np2[np.arange(neg_advs_np2.shape[0]), env_action2] = a2_neg_advs_v
                    neg_advs2 = mx.nd.array(neg_advs_np2, ctx=q_ctx)

                    value_grads1 = mx.nd.array(config.vf_wt * a1_neg_advs_v[:, np.newaxis],
                                               ctx=q_ctx)
                    value_grads2 = mx.nd.array(config.vf_wt * a2_neg_advs_v[:, np.newaxis],
                                               ctx=q_ctx)

                    env_ob1 = []
                    for i in range(num_envs):
                        env_ob1.extend(envs[i].ob1)
                    a1_AC.model.reshape([('data', (len(env_ob1), 74))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(env_ob1, ctx=q_ctx)], label=None)
                    a1_AC.model.forward(data_batch, is_train=True)
                    _, episode_value1, _, episode_policy1 = a1_AC.model.get_outputs()
                    a1_AC.model.backward(out_grads=[neg_advs1, value_grads1])
                    a1_AC.model.update()

                    env_ob2 = []
                    for i in range(num_envs):
                        env_ob2.extend(envs[i].ob2)
                    a2_AC.model.reshape([('data', (len(env_ob2), 74))])
                    data_batch = mx.io.DataBatch(data=[mx.nd.array(env_ob2, ctx=q_ctx)], label=None)
                    a2_AC.model.forward(data_batch, is_train=True)
                    _, episode_value2, _, episode_policy2 = a2_AC.model.get_outputs()
                    a2_AC.model.backward(out_grads=[neg_advs2, value_grads2])
                    a2_AC.model.update()

                    for i in range(num_envs):
                        envs[i].clear()

                    training_steps += 1
                    episode_advs1 += np.mean(a1_neg_advs_v)
                    episode_advs2 += np.mean(a2_neg_advs_v)
                    episode_values1 += np.mean(episode_value1.asnumpy())
                    episode_values2 += np.mean(episode_value2.asnumpy())
                    episode_policys1 += np.max(episode_policy1[0].asnumpy())
                    episode_policys2 += np.max(episode_policy2[0].asnumpy())
                    t = 0

                all_done = np.all([envs[i].done for i in range(num_envs)])
                t += 1

            steps_left -= episode_step
            epoch_reward += episode_reward
            time_episode_end = time.time()
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, Collisions:%f, fps:%f, Advs:%f/%f/%d, Value:%f/%f, Policys:%f/%f" \
                       % (
                           epoch, episode, steps_left, episode_step, steps_per_epoch,
                           np.mean(episode_reward), np.mean(collisions),
                           episode_step / (time_episode_end - time_episode_start),
                           episode_advs1 / training_steps, episode_advs2 / training_steps, training_steps,
                           episode_values1 / training_steps, episode_values2 / training_steps,
                           episode_policys1 / training_steps, episode_policys2 / training_steps)
            logging.info(info_str)
            print info_str

        end = time.time()
        fps = steps_per_epoch / (end - start)
        a1_AC.save_params(epoch)
        a2_AC.save_params(epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))


if __name__ == '__main__':
    main()
