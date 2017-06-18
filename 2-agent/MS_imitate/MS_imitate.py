from hunterworld_v2 import HunterWorld
from ple_v7 import PLE
import argparse
from utils import *
from Qnet import *
from replay_memory import ReplayMemory
from config import Config
import logging

dir = os.path.dirname(__file__)
os.path.join(os.path.realpath('..'))
file_name = os.path.basename(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--t-max', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=100000)
    parser.add_argument('--testing', type=int, default=0)
    parser.add_argument('--continue-training', type=int, default=4)
    parser.add_argument('--epoch-num', type=int, default=20)
    parser.add_argument('--start-epoch', type=int, default=20)
    parser.add_argument('--testing-epoch', type=int, default=0)
    parser.add_argument('--save-log', type=str, default='log')
    parser.add_argument('--signal-num', type=int, default=4)
    parser.add_argument('--toxin', type=int, default=0)
    parser.add_argument('--a1-AC-folder', type=str, default='basic/a1_Qnet')
    parser.add_argument('--a2-AC-folder', type=str, default='basic/a2_Qnet')
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--replay-start-size', type=int, default=50000)
    parser.add_argument('--decay-rate', type=int, default=50000)
    parser.add_argument('--replay-memory-size', type=int, default=1000000)
    parser.add_argument('--eps-min', type=float, default=0.1)

    args = parser.parse_args()
    config = Config(args)
    t_max = args.t_max
    q_ctx = config.ctx
    steps_per_epoch = args.steps_per_epoch
    np.random.seed(args.seed)
    start_epoch = args.start_epoch
    testing_epoch = args.testing_epoch
    save_log = args.save_log
    epoch_num = args.epoch_num
    epoch_range = range(epoch_num)
    signal_num = args.signal_num
    toxin = args.toxin
    a1_Qnet_folder = args.a1_AC_folder
    a2_Qnet_folder = args.a2_AC_folder

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

    rewards = {
        "positive": 1.0,
        "negative": -1.0,
        "tick": -0.002,
        "loss": -2.0,
        "win": 2.0
    }

    game = HunterWorld(width=256, height=256, num_preys=10, draw=False,
                       num_hunters=2, num_toxins=toxin)
    env = PLE(game, fps=30, force_fps=True, display_screen=False, reward_values=rewards,
              resized_rows=80, resized_cols=80, num_steps=3)

    action_set = env.get_action_set()
    action_map1 = []
    for action in action_set[0].values():
        action_map1.append(action)

    action_map2 = []
    for action in action_set[1].values():
        action_map2.append(action)
    action_num = len(action_map1)

    replay_memory1 = ReplayMemory(state_dim=(74,),
                                  history_length=history_length,
                                  memory_size=replay_memory_size,
                                  replay_start_size=replay_start_size, state_dtype='float32')


    a1_target1 = Qnetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=1, dir=dir,
                          folder=a1_Qnet_folder)
    a1_target32 = Qnetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=32, dir=dir,
                           folder=a1_Qnet_folder)
    Qnet1 = Qnetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=True, batch_size=32, dir=dir,
                     folder=a1_Qnet_folder)

    a2_target1 = Qnetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=1, dir=dir,
                          folder=a2_Qnet_folder)
    a2_target32 = Qnetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=False, batch_size=32, dir=dir,
                           folder=a2_Qnet_folder)
    Qnet2 = Qnetwork(actions_num=action_num, q_ctx=q_ctx, isTrain=True, batch_size=32, dir=dir,
                     folder=a2_Qnet_folder)

    training_steps = 0
    total_steps = 0
    if testing:
        env.force_fps = False
        env.game.draw = True
        env.display_screen = True
        Qnet1.load_params(testing_epoch)
        Qnet2.load_params(testing_epoch)
    elif continue_training:
        epoch_range = range(start_epoch, epoch_num + start_epoch)
        Qnet1.load_params(start_epoch - 1)
        Qnet2.load_params(start_epoch - 1)
        logging_config(logging, dir, save_log, file_name)
    else:
        logging_config(logging, dir, save_log, file_name)

    copyTargetQNetwork(Qnet1.model, a1_target1.model)
    copyTargetQNetwork(Qnet1.model, a1_target32.model)
    copyTargetQNetwork(Qnet2.model, a2_target1.model)
    copyTargetQNetwork(Qnet2.model, a2_target32.model)

    logging.info('args=%s' % args)
    logging.info('config=%s' % config.__dict__)
    print_params(logging, Qnet1.model)
    print_params(logging, Qnet2.model)

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
            next_ob = env.get_states()
            while not env.game_over():
                if replay_memory1.size >= history_length and replay_memory1.size > replay_start_size:
                    do_exploration = (np.random.rand() < eps_curr)
                    eps_curr = max(eps_curr - eps_decay, eps_min)
                    if do_exploration:
                        action1 = np.random.randint(action_num)
                        action2 = np.random.randint(action_num)
                    else:
                        current_state1 = next_ob[0].reshape(1, 74)
                        current_state2 = next_ob[1].reshape(1, 74)
                        state1 = nd.array(current_state1.reshape((1,) + current_state1.shape),
                                          ctx=q_ctx)
                        state2 = nd.array(current_state2.reshape((1,) + current_state2.shape),
                                          ctx=q_ctx)
                        a1_target1.model.forward(mx.io.DataBatch([state1], []))
                        a2_target1.model.forward(mx.io.DataBatch([state2], []))
                        q_value1 = a1_target1.model.get_outputs()[0].asnumpy()[0]
                        q_value2 = a2_target1.model.get_outputs()[0].asnumpy()[0]
                        action1 = numpy.argmax(q_value1)
                        action2 = numpy.argmax(q_value2)
                        episode_q_value += q_value1[action1]
                        episode_q_value += q_value2[action2]
                        episode_action_step += 1
                else:
                    action1 = np.random.randint(action_num)
                    action2 = np.random.randint(action_num)

                next_ob, reward, terminal_flag = env.act([action_map1[action1], action_map2[action2]])
                replay_memory1.append(next_ob[0], action1, reward[0], terminal_flag)

                total_steps += 1
                sum_reward = sum(reward)
                episode_reward += sum_reward
                if sum_reward < 0:
                    collisions += 1
                episode_step += 1

                if total_steps % update_interval == 0 and replay_memory1.size > replay_start_size:
                    training_steps += 1

                    state_batch1, actions1, rewards1, nextstate_batch1, terminate_flags1 = replay_memory1.sample(
                        batch_size=minibatch_size)
                    state_batch2, actions2, rewards2, nextstate_batch2, terminate_flags2 = replay_memory1.sample(
                        batch_size=minibatch_size)

                    state_batch1 = nd.array(state_batch1, ctx=q_ctx)
                    actions_batch1 = nd.array(actions1, ctx=q_ctx)
                    reward_batch1 = nd.array(rewards1, ctx=q_ctx)
                    terminate_flags1 = nd.array(terminate_flags1, ctx=q_ctx)

                    state_batch2 = nd.array(state_batch2, ctx=q_ctx)
                    actions_batch2 = nd.array(actions2, ctx=q_ctx)
                    reward_batch2 = nd.array(rewards2, ctx=q_ctx)
                    terminate_flags2 = nd.array(terminate_flags2, ctx=q_ctx)

                    a1_target32.model.forward(mx.io.DataBatch([nd.array(nextstate_batch1, ctx=q_ctx)], []))
                    Qvalue1 = a1_target32.model.get_outputs()[0]

                    y_batch1 = reward_batch1 + nd.choose_element_0index(Qvalue1, nd.argmax_channel(Qvalue1)) * (
                        1.0 - terminate_flags1) * discount

                    Qnet1.model.forward(mx.io.DataBatch([state_batch1, actions_batch1, y_batch1],
                                                        []), is_train=True)
                    Qnet1.model.backward()
                    Qnet1.model.update()

                    a2_target32.model.forward(mx.io.DataBatch([nd.array(nextstate_batch2, ctx=q_ctx)], []))
                    Qvalue2 = a2_target32.model.get_outputs()[0]

                    y_batch2 = reward_batch2 + nd.choose_element_0index(Qvalue2, nd.argmax_channel(Qvalue2)) * (
                        1.0 - terminate_flags2) * discount

                    Qnet2.model.forward(mx.io.DataBatch([state_batch2, actions_batch2, y_batch2],
                                                        []), is_train=True)
                    Qnet2.model.backward()
                    Qnet2.model.update()

                    if training_steps % 10 == 0:
                        loss1 = 0.5 * nd.square(
                            nd.choose_element_0index(Qnet1.model.get_outputs()[0], actions_batch1) - y_batch1)
                        loss2 = 0.5 * nd.square(
                            nd.choose_element_0index(Qnet2.model.get_outputs()[0], actions_batch2) - y_batch2)
                        episode_loss += nd.sum(loss1).asnumpy()
                        episode_loss += nd.sum(loss2).asnumpy()
                        episode_update_step += 1

                    if training_steps % freeze_interval == 0:
                        copyTargetQNetwork(Qnet1.model, a1_target1.model)
                        copyTargetQNetwork(Qnet1.model, a1_target32.model)
                        copyTargetQNetwork(Qnet2.model, a2_target1.model)
                        copyTargetQNetwork(Qnet2.model, a2_target32.model)

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
                                                  episode_update_step * 10)
            if episode_action_step > 0:
                info_str += ", Avg Q Value:%f/%d " % (episode_q_value / episode_action_step,
                                                      episode_action_step)

            if episode % 1 == 0:
                logging.info(info_str)
                print info_str

        end = time.time()
        fps = steps_per_epoch / (end - start)
        Qnet1.save_params(epoch)
        Qnet2.save_params(epoch)
        logging.info("Epoch:%d, FPS:%f, Avg Reward: %f/%d"
                     % (epoch, fps, epoch_reward / float(episode), episode))


if __name__ == '__main__':
    main()
