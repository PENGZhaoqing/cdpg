from io import open
import matplotlib.pyplot as plt
import numpy as np
import re


class NetworkGraph(object):
    def __init__(self, file_name):
        self.steps = []
        self.collision = []
        self.avg_adv = []
        self.avg_Q_value = []
        self.filename = file_name
        self.policies = []
        self.on_rewards = []
        self.fps = []
        self.reward = []

    def load(self, offline=False):
        if offline:
            with open(self.filename, 'r') as text_file:
                lines = text_file.readlines()
                for line in lines:
                    columns = line.split(',')
                    if len(columns) == 10:
                        self.reward.append(columns[4].split(':')[1])
                        self.fps.append(columns[5].split(":")[1])
                        self.collision.append(columns[7].split(':')[1].split('/')[1])
                        self.avg_Q_value.append(columns[9].split(':')[1].split('/')[0])
        else:
            with open(self.filename, 'r') as text_file:
                lines = text_file.readlines()
                for line in lines:
                    columns = line.split(',')
                    if len(columns) == 10:
                        self.reward.append(columns[4].split(':')[1])
                        self.collision.append(columns[5].split(":")[1])
                        self.fps.append(columns[6].split(":")[1])
                        self.avg_Q_value.append(columns[8].split(':')[1].split('/')[0])
                        self.policies.append(columns[9].split(':')[1].split('/')[0])
        self.steps = np.array(self.steps, dtype=int)
        self.reward = np.array(self.reward, dtype=float)
        self.avg_adv = np.array(self.avg_adv, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)
        self.policies = np.array(self.policies, dtype=float)
        self.fps = np.array(self.fps, dtype=float)
        self.collision = np.array(self.collision, dtype=float)

        with open(self.filename, 'r') as text_file:
            lines = text_file.read().replace('\n', '')
            tmp = re.findall('\[.*?\]', lines)
            for line in tmp:
                rewards = line.replace('[', '').replace(']', '').split('. ')
                self.on_rewards.append(rewards)
        self.on_rewards = np.array(self.on_rewards, dtype=float)


basic_IL_offline = NetworkGraph("2-agents/offline/IL_offline.py_06_20_03_43_adam-0.002.log")
basic_IL_offline.load(True)
basic_GL_offline = NetworkGraph("2-agents/offline/GL_offline.py_06_19_21_43_adam-0.002.log")
basic_GL_offline.load(True)
basic_PS_offline = NetworkGraph("2-agents/offline/PS_offline.py_06_20_00_33_adam-0.002.log")
basic_PS_offline.load(True)
basic_ML_offline = NetworkGraph("2-agents/offline/ML_offline.py_06_21_21_09_adam-0.002.log")
basic_ML_offline.load(True)
basic_MS_imitate = NetworkGraph("2-agents/offline/MS_imitate.py_06_20_08_34_adam-0.002.log")
basic_MS_imitate.load(True)

basic_IL_online = NetworkGraph("2-agents/online/IL_online.py_t_max1_06_19_11_25_adam-0.002.log")
basic_IL_online.load()
basic_GL_online = NetworkGraph("2-agents/online/GL_online.py_t_max_1_06_21_20_06_adam-0.002.log")
basic_GL_online.load()
basic_PS_online = NetworkGraph("2-agents/online/PS_online.py_t_max_1_06_22_09_40_adam-0.002.log")
basic_PS_online.load()
basic_ML_online = NetworkGraph("2-agents/online/ML_online.py_t_max_1_06_18_17_44_adam-0.002.log")
basic_ML_online.load()

toxin_IL_offline = NetworkGraph("2-agent-with-toxin/offline/IL_offline.py_06_20_06_04_adam-0.002.log")
toxin_IL_offline.load(True)
toxin_GL_offline = NetworkGraph("2-agent-with-toxin/offline/GL_offline.py_06_19_23_05_adam-0.002.log")
toxin_GL_offline.load(True)
toxin_PS_offline = NetworkGraph("2-agent-with-toxin/offline/PS_offline.py_06_20_02_05_adam-0.002.log")
toxin_PS_offline.load(True)
toxin_ML_offline = NetworkGraph("2-agent-with-toxin/offline/ML_offline.py_06_22_00_20_adam-0.002.log")
toxin_ML_offline.load(True)
toxin_MS_imitate = NetworkGraph("2-agent-with-toxin/offline/MS_imitate.py_06_20_10_21_adam-0.002.log")
toxin_MS_imitate.load(True)

toxin_IL_online = NetworkGraph("2-agent-with-toxin/online/IL_online.py_t_max1_06_19_12_53_adam-0.002.log")
toxin_IL_online.load()
toxin_GL_online = NetworkGraph("2-agent-with-toxin/online/GL_online.py_t_max_1_06_22_02_12_adam-0.002.log")
toxin_GL_online.load()
toxin_PS_online = NetworkGraph("2-agent-with-toxin/online/PS_online.py_t_max_1_06_22_11_09_adam-0.002.log")
toxin_PS_online.load()
toxin_ML_online = NetworkGraph("2-agent-with-toxin/online/ML_online.py_t_max_1_06_18_15_57_adam-0.002.log")
toxin_ML_online.load()


def smooth_offline(data):
    a = len(data) % 2
    if a != 0:
        data = data[0:-a].reshape(-1, 2).mean(axis=1)
    else:
        data = data.reshape(-1, 2).mean(axis=1)
    data2 = data[0:80 * 24]
    return data2.reshape(-1, 24)
    # return data1.reshape(-1, 48)


def smooth_online(data):
    return data.reshape(80, -1, 32).mean(axis=1)


def last_20_epoch_offline(data):
    return data[-40 * 48:]


def mean_median_std(object, toxin=False):
    reward = last_20_epoch_offline(object.reward)
    fps = object.fps
    result = []
    result.append(np.mean(reward))
    result.append(np.std(reward))
    if toxin:
        collision = last_20_epoch_offline(object.collision)
        result.append(np.mean(collision))
        result.append(np.std(collision))
        result.append(np.mean(reward) / np.mean(collision))
    result.append(np.mean(fps))
    result.append(np.std(fps))
    return ['{:.2f}'.format(i) for i in result]

def last_20_epoch_online(data):
    return data[-80:, :]

def mean_median_std_online(object, toxin=False):
    reward = last_20_epoch_online(object.on_rewards)
    fps = object.fps
    result = []
    result.append(np.mean(reward))
    result.append(np.std(reward))
    if toxin:
        collision = object.collision[-80:]
        result.append(np.mean(collision))
        result.append(np.std(collision))
        result.append(np.mean(reward) / np.mean(collision))
    result.append(np.mean(fps))
    result.append(np.std(fps))
    return ['{:.2f}'.format(i) for i in result]

print "offline"
print 'IL:' + str(mean_median_std(basic_IL_offline)) + str(mean_median_std(toxin_IL_offline, toxin=True))
print 'GL:' + str(mean_median_std(basic_GL_offline)) + str(mean_median_std(toxin_GL_offline, toxin=True))
print 'PS:' + str(mean_median_std(basic_PS_offline)) + str(mean_median_std(toxin_PS_offline, toxin=True))
print 'MS:' + str(mean_median_std(basic_MS_imitate)) + str(mean_median_std(toxin_MS_imitate, toxin=True))
print 'ML:' + str(mean_median_std(basic_ML_offline)) + str(mean_median_std(toxin_ML_offline, toxin=True))

print "online"
print 'IL:' + str(mean_median_std_online(basic_IL_online)) + str(mean_median_std_online(toxin_IL_online, toxin=True))
print 'GL:' + str(mean_median_std_online(basic_GL_online)) + str(mean_median_std_online(toxin_GL_online, toxin=True))
print 'PS:' + str(mean_median_std_online(basic_PS_online)) + str(mean_median_std_online(toxin_PS_online, toxin=True))
print 'ML:' + str(mean_median_std_online(basic_ML_online)) + str(mean_median_std_online(toxin_ML_online, toxin=True))
