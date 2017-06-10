import argparse
import logging
from Env_two import Env
from A3c_network import *
from config import Config
from utils import *


def _2d_list(n):
    return [[] for _ in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--t-max', type=int, default=1)
    parser.add_argument('--save-pre', default='checkpoints')
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logging.info-every', type=int, default=1)

    args = parser.parse_args()
    config = Config(args)
    np.random.seed(args.seed)

    if not os.path.exists('a1_A3c'):
        os.makedirs('a1_A3c')
    if not os.path.exists('a2_A3c'):
        os.makedirs('a2_A3c')

    envs = []
    for i in range(args.num_envs):
        envs.append(Env(i))

    # logging_config(logging, 'log', 'test_com_tmax_' + str(args.t_max))

    # envs[0].ple.force_fps = False
    # envs[0].game.draw = True
    # envs[0].ple.display_screen = True
    a1_A3c_file = 'a1_A3c/network-dqn_mx0019.params'
    a2_A3c_file = 'a2_A3c/network-dqn_mx0019.params'
    a1_comNet_file = 'a1_comNet/network-dqn_mx0019.params'
    a2_comNet_file = 'a2_comNet/network-dqn_mx0019.params'

    action_num = 4
    a1_A3c = A3c(state_dim=74, signal_num=4, act_space=action_num, config=config)
    a2_A3c = A3c(state_dim=74, signal_num=4, act_space=action_num, config=config)

    a1_comNet = ComNet(state_dim=74, signal_num=4, config=config)
    a2_comNet = ComNet(state_dim=74, signal_num=4, config=config)

    a1_A3c.model.load_params(a1_A3c_file)
    a2_A3c.model.load_params(a2_A3c_file)
    a1_comNet.model.load_params(a1_comNet_file)
    a2_comNet.model.load_params(a2_comNet_file)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(logging, a1_A3c.model)
    print_params(logging, a1_comNet.model)

    action_set = envs[0].ple.get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)

    t_max = args.t_max
    num_envs = len(envs)
    epoch_num = 20
    steps_per_epoch = 100000

    for epoch in range(epoch_num):
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
            training_steps = 0
            sample_num = 0.0
            episode_values1 = 0
            episode_values2 = 0
            episode_policys1 = 0
            episode_policys2 = 0
            signal_statis = [[], [], [], []]

            while not all_done:

                signal1 = a1_comNet.forward(step_ob2, is_train=False)[0]
                signal2 = a2_comNet.forward(step_ob1, is_train=False)[0]

                _, step_value1, _, step_policy1 = a1_A3c.forward(step_ob1, signal1, is_train=False)
                _, step_value2, _, step_policy2 = a2_A3c.forward(step_ob2, signal2,
                                                                 is_train=False)
                step_policy1 = step_policy1.asnumpy()
                step_value1 = step_value1.asnumpy()
                step_policy2 = step_policy2.asnumpy()
                step_value2 = step_value2.asnumpy()

                us1 = np.random.uniform(size=step_policy1.shape[0])[:, np.newaxis]
                step_action1 = (np.cumsum(step_policy1, axis=1) > us1).argmax(axis=1)

                us2 = np.random.uniform(size=step_policy2.shape[0])[:, np.newaxis]
                step_action2 = (np.cumsum(step_policy2, axis=1) > us2).argmax(axis=1)

                step_policy2_test = a2_A3c.forward(np.zeros(np.array(step_ob2).shape), signal2,
                                                                 is_train=False)[3].asnumpy()
                us2_test = np.random.uniform(size=step_policy2_test.shape[0])[:, np.newaxis]
                step_action2_test = (np.cumsum(step_policy2_test, axis=1) > us2_test).argmax(axis=1)

                signal2 = signal2.asnumpy()
                for i in range(len(step_action2_test)):
                    if step_action2_test[i] == 0:
                        signal_statis[0].append(signal2[i])
                    elif step_action2_test[i] == 1:
                        signal_statis[1].append(signal2[i])
                    elif step_action2_test[i] == 2:
                        signal_statis[2].append(signal2[i])
                    elif step_action2_test[i] == 3:
                        signal_statis[3].append(signal2[i])

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

                    env_ob1 = []
                    env_ob2 = []
                    for i in range(num_envs):
                        env_ob1.extend(envs[i].ob1)
                        env_ob2.extend(envs[i].ob2)

                    signal1 = a1_comNet.forward(env_ob2, is_train=False)[0]
                    signal2 = a2_comNet.forward(env_ob1, is_train=False)[0]

                    _, episode_value1, _, episode_policy1 = a1_A3c.forward(env_ob1, signal1, is_train=False)
                    _, episode_value2, _, episode_policy2 = a2_A3c.forward(env_ob2, signal2, is_train=False)

                    for i in range(num_envs):
                        envs[i].clear()

                    training_steps += 1
                    if training_steps % 10 == 0:
                        sample_num += 1
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
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, Collisions:%f, fps:%f, Value:%f/%f, Policys:%f/%f" \
                       % (
                           epoch, episode, steps_left, episode_step, steps_per_epoch,
                           np.mean(episode_reward), np.mean(collisions),
                           episode_step / (time_episode_end - time_episode_start),
                           episode_values1 / sample_num, episode_values2 / sample_num,
                           episode_policys1 / sample_num, episode_policys2 / sample_num)

            print str(np.mean(signal_statis[0], axis=0))
            print str(np.mean(signal_statis[1], axis=0))
            print str(np.mean(signal_statis[2], axis=0))
            print str(np.mean(signal_statis[3], axis=0))

            logging.info(info_str)
            print info_str

        end = time.time()
        fps = steps_per_epoch / (end - start)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, np.mean(epoch_reward) / float(episode), episode))


if __name__ == '__main__':
    main()
