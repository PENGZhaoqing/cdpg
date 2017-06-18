import numpy as np
from hunterworld_v2 import HunterWorld
from ple_v7 import PLE
from utils import *
import matplotlib.pyplot as plt
from central_network import Qnetwork, copyTargetQNetwork
from replay_memory import ReplayMemory
import mxnet as mx
import mxnet.ndarray as nd
import numpy
import argparse
from config import Config
import logging

dir = os.path.dirname(__file__)
os.path.join(os.path.realpath('..'))
file_name = os.path.basename(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--t-max', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=100000)
    parser.add_argument('--testing', type=int, default=0)
    parser.add_argument('--continue-training', type=int, default=0)
    parser.add_argument('--epoch-num', type=int, default=40)
    parser.add_argument('--start-epoch', type=int, default=20)
    parser.add_argument('--testing-epoch', type=int, default=3)
    parser.add_argument('--save-log', type=str, default='basic/log')
    parser.add_argument('--signal-num', type=int, default=4)
    parser.add_argument('--toxin', type=int, default=0)
    parser.add_argument('--a1-Qnet-folder', type=str, default='basic/a1_Qnet')
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--replay-start-size', type=int, default=50000)
    parser.add_argument('--decay-rate', type=int, default=500000)
    parser.add_argument('--replay-memory-size', type=int, default=1000000)
    parser.add_argument('--eps-min', type=float, default=0.05)

    rewards = {
        "positive": 1.0,
        "negative": -1.0,
        "tick": -0.002,
        "loss": -2.0,
        "win": 2.0
    }

    args = parser.parse_args()
    config = Config(args)
    q_ctx = config.ctx
    steps_per_epoch = args.steps_per_epoch
    np.random.seed(args.seed)
    start_epoch = args.start_epoch
    testing_epoch = args.testing_epoch
    save_log = args.save_log
    epoch_num = args.epoch_num
    epoch_range = range(epoch_num)
    toxin = args.toxin
    a1_Qnet_folder = args.a1_Qnet_folder

    freeze_interval = 10000
    update_interval = 5
    replay_memory_size = args.replay_memory_size
    discount = 0.99
    replay_start_size = args.replay_start_size
    history_length = 1
    eps_start = args.eps_start
    eps_min = args.eps_min
    eps_decay = (eps_start - eps_min) / args.decay_rate
    eps_curr = eps_start
    freeze_interval /= update_interval
    minibatch_size = 32

    testing = args.testing
    testing = True if testing == 1 else False
    continue_training = args.continue_training
    continue_training = True if continue_training == 1 else False

    game = HunterWorld(width=256, height=256, num_preys=10, draw=False,
                       num_hunters=2, num_toxins=toxin)

    env = PLE(game, fps=30, force_fps=True, display_screen=False,
              reward_values=rewards,
              resized_rows=80, resized_cols=80, num_steps=2)

    replay_memory = ReplayMemory(state_dim=(148,), action_dim=(2,),
                                 history_length=history_length,
                                 memory_size=replay_memory_size,
                                 replay_start_size=replay_start_size, state_dtype='float32')

    action_set = env.get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)
    action_num = len(action_map1)

    target1 = Qnetwork(actions_num=8, q_ctx=q_ctx, isTrain=False, batch_size=1, dir=dir,
                       folder=a1_Qnet_folder)
    target32 = Qnetwork(actions_num=8, q_ctx=q_ctx, isTrain=False, batch_size=32, dir=dir,
                        folder=a1_Qnet_folder)
    Qnet = Qnetwork(actions_num=8, q_ctx=q_ctx, isTrain=True, batch_size=32, dir=dir,
                    folder=a1_Qnet_folder)

    if testing:
        env.force_fps = False
        env.game.draw = True
        env.display_screen = True
        Qnet.load_params(testing_epoch)
    elif continue_training:
        epoch_range = range(start_epoch, epoch_num + start_epoch)
        Qnet.load_params(start_epoch - 1)
        logging_config(logging, dir, save_log, file_name)
    else:
        logging_config(logging, dir, save_log, file_name)

    copyTargetQNetwork(Qnet.model, target1.model)
    copyTargetQNetwork(Qnet.model, target32.model)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(logging, Qnet.model)

    training_steps = 0
    total_steps = 0
    for epoch in epoch_range:
        steps_left = steps_per_epoch
        episode = 0
        epoch_reward = 0
        start = time.time()
        env.reset_game()
        while steps_left > 0:
            episode += 1
            episode_loss = 0.0
            episode_q_value = 0.0
            episode_update_step = 0
            episode_action_step = 0
            episode_reward = 0
            episode_step = 0
            collisions = 0.0
            time_episode_start = time.time()
            env.reset_game()
            while not env.game_over():
                if replay_memory.size >= history_length and replay_memory.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action1 = np.random.randint(action_num)
                        action2 = np.random.randint(action_num)
                    else:
                        current_state = replay_memory.latest_slice()
                        state = nd.array(current_state.reshape((1,) + current_state.shape),
                                         ctx=q_ctx)
                        target1.model.forward(mx.io.DataBatch([state], []))
                        q_value = target1.model.get_outputs()[0].asnumpy()[0]
                        action1 = numpy.argmax(q_value[0:4])
                        action2 = numpy.argmax(q_value[4:8])
                        episode_q_value += q_value[action1]
                        episode_q_value += q_value[action2 + 4]
                        episode_action_step += 1
                else:
                    action1 = np.random.randint(action_num)
                    action2 = np.random.randint(action_num)

                next_ob, reward, terminal_flag = env.act([action_map1[action1], action_map2[action2]])

                reward = np.sum(reward)
                replay_memory.append(np.array(next_ob).flatten(), [action1, action2], reward, terminal_flag)

                total_steps += 1
                episode_reward += reward
                if reward < 0:
                    collisions += 1
                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory.size > replay_start_size:
                    training_steps += 1

                    state_batch, actions, rewards, nextstate_batch, terminate_flags = replay_memory.sample(
                        batch_size=minibatch_size)
                    state_batch = nd.array(state_batch, ctx=q_ctx)
                    actions_batch1 = nd.array(actions[:, 0], ctx=q_ctx)
                    actions_batch2 = nd.array(actions[:, 1], ctx=q_ctx)

                    target32.model.forward(mx.io.DataBatch([nd.array(nextstate_batch, ctx=q_ctx)], []))
                    Qvalue = target32.model.get_outputs()[0].asnumpy()

                    y_batch1 = rewards + np.max(Qvalue[:, 0:4], axis=1) * (1.0 - terminate_flags) * discount
                    y_batch2 = rewards + np.max(Qvalue[:, 4:8], axis=1) * (1.0 - terminate_flags) * discount

                    y_batch1 = nd.array(y_batch1, ctx=q_ctx)
                    y_batch2 = nd.array(y_batch2, ctx=q_ctx)

                    Qnet.model.forward(
                        mx.io.DataBatch([state_batch, actions_batch1, y_batch1, actions_batch2 + 4, y_batch2],
                                        []), is_train=True)
                    Qnet.model.backward()
                    Qnet.model.update()

                    if training_steps % 10 == 0:
                        out = Qnet.model.get_outputs()
                        loss1 = 0.5 * nd.square(
                            nd.choose_element_0index(out[0], actions_batch1) - y_batch1)
                        loss2 = 0.5 * nd.square(
                            nd.choose_element_0index(out[1], actions_batch2) - y_batch2)
                        episode_loss += nd.sum(loss1).asnumpy()
                        episode_loss += nd.sum(loss2).asnumpy()
                        episode_update_step += 1

                    if training_steps % freeze_interval == 0:
                        copyTargetQNetwork(Qnet.model, target1.model)
                        copyTargetQNetwork(Qnet.model, target32.model)

            steps_left -= episode_step
            time_episode_end = time.time()
            epoch_reward += episode_reward
            info_str = "Epoch:%d, Episode:%d, Steps Left:%d/%d/%d, Reward:%f, fps:%f, Exploration:%f" \
                       % (epoch, episode, steps_left, episode_step, steps_per_epoch, episode_reward,
                          episode_step / (time_episode_end - time_episode_start), eps_curr)

            info_str += ", Collision:%f/%d " % (collisions / episode_step,
                                                collisions)

            if episode_update_step > 0:
                info_str += ", Avg Loss:%f/%d" % (episode_loss / episode_update_step,
                                                  episode_update_step)
            if episode_action_step > 0:
                info_str += ", Avg Q Value:%f/%d " % (episode_q_value / episode_action_step,
                                                      episode_action_step)

            if episode % 1 == 0:
                print info_str

        end = time.time()
        fps = steps_per_epoch / (end - start)
        Qnet.save_params(epoch)
        print "Epoch:%d, FPS:%f, Avg Reward: %f/%d" % (epoch, fps, epoch_reward / float(episode), episode)


if __name__ == '__main__':
    main()
