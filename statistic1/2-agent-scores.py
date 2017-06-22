from io import open
import matplotlib.pyplot as plt
import numpy as np
import re


class NetworkGraph(object):
    def __init__(self, file_name):
        self.steps = []
        self.reward = []
        self.collision = []
        self.avg_adv = []
        self.avg_Q_value = []
        self.filename = file_name
        self.policies = []
        self.mean_Qvalue = None
        self.mean_reward = None
        self.mean_steps = None
        self.mean_collision = None
        self.mean_loss = None
        self.rewards = []

    def load(self):
        with open(self.filename, 'r') as text_file:
            lines = text_file.readlines()
            for line in lines:
                columns = line.split(',')
                if len(columns) == 10:
                    self.steps.append(columns[3].split('/')[1])
                    self.reward.append(columns[4].split(':')[1])
                    self.avg_adv.append(columns[7].split(':')[1].split('/')[0])
                    self.avg_Q_value.append(columns[8].split(':')[1].split('/')[0])
                    self.policies.append(columns[9].split(':')[1].split('/')[0])
        self.steps = np.array(self.steps, dtype=int)
        self.reward = np.array(self.reward, dtype=float)
        self.avg_adv = np.array(self.avg_adv, dtype=float)
        self.avg_Q_value = np.array(self.avg_Q_value, dtype=float)
        self.policies = np.array(self.policies, dtype=float)

        with open(self.filename, 'r') as text_file:
            lines = text_file.read().replace('\n', '')
            tmp = re.findall('\[.*?\]', lines)
            for line in tmp:
                rewards = line.replace('[', '').replace(']', '').split('. ')
                self.rewards.append(rewards)
        self.rewards = np.array(self.rewards, dtype=float)


def range_x(data):
    return np.array(range(len(data))) * 0.5


def confidence_band_offline(data1, data2, data3, data4, data5):
    x1 = range_x(data1)
    x2 = range_x(data2)
    x3 = range_x(data3)
    x4 = range_x(data4)
    x5 = range_x(data5)

    mean1, std1 = mean_std(data1)
    mean2, std2 = mean_std(data2)
    mean3, std3 = mean_std(data3)
    mean4, std4 = mean_std(data4)
    mean5, std5 = mean_std(data5)

    # # greeen
    plt.plot(x1, mean1, 'k', color='#009E73', label="IL-offline")
    plt.fill_between(x1, mean1 - std1, mean1 + std1,
                     alpha=0.1, edgecolor='#009E73', facecolor='#009E73',
                     linewidth=1)

    # # pink
    plt.plot(x2, mean2, 'k', color='#c46ea0', label="GL-offline")
    plt.fill_between(x2, mean2 - std2, mean2 + std2,
                     alpha=0.1, edgecolor='#c46ea0', facecolor='#c46ea0',
                     linewidth=1)

    # blue
    plt.plot(x3, mean3, 'k', color='#0B77B5', label="PS-offline")
    plt.fill_between(x3, mean3 - std3, mean3 + std3,
                     alpha=0.1, edgecolor='#0B77B5', facecolor='#0B77B5',
                     linewidth=1)

    # red
    plt.plot(x4, mean4, 'k', color='#ff0066', label="ML-offline")
    plt.fill_between(x4, mean4 - std4, mean4 + std4,
                     alpha=0.1, edgecolor='#ff0066', facecolor='#ff0066',
                     linewidth=1)

    # # orange
    plt.plot(x5, mean5, 'k', color='#ff9900', label="MS")
    plt.fill_between(x5, mean5 - std5, mean5 + std5,
                     alpha=0.1, edgecolor='#ff9900', facecolor='#ff9900',
                     linewidth=1)

    plt.legend(loc=2, labelspacing=0, fancybox=True, framealpha=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 40])
    axes.set_ylim([-50, 450])
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def confidence_band_online(data1, data2, data3, data4):
    x1 = range_x(data1)
    x2 = range_x(data2)
    x3 = range_x(data3)
    x4 = range_x(data4)

    mean1, std1 = mean_std(data1)
    mean2, std2 = mean_std(data2)
    mean3, std3 = mean_std(data3)
    mean4, std4 = mean_std(data4)

    # # greeen
    plt.plot(x1, mean1, 'k', color='#009E73', label="IL-online")
    plt.fill_between(x1, mean1 - std1, mean1 + std1,
                     alpha=0.1, edgecolor='#009E73', facecolor='#009E73',
                     linewidth=1)

    # # pink
    plt.plot(x2, mean2, 'k', color='#c46ea0', label="GL-online")
    plt.fill_between(x2, mean2 - std2, mean2 + std2,
                     alpha=0.1, edgecolor='#c46ea0', facecolor='#c46ea0',
                     linewidth=1)

    # blue
    plt.plot(x3, mean3, 'k', color='#0B77B5', label="PS-online")
    plt.fill_between(x3, mean3 - std3, mean3 + std3,
                     alpha=0.1, edgecolor='#0B77B5', facecolor='#0B77B5',
                     linewidth=1)

    # red
    plt.plot(x4, mean4, 'k', color='#ff0066', label="ML-online")
    plt.fill_between(x4, mean4 - std4, mean4 + std4,
                     alpha=0.1, edgecolor='#ff0066', facecolor='#ff0066',
                     linewidth=1)

    plt.legend(loc=2, labelspacing=0, fancybox=True, framealpha=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 40])
    axes.set_ylim([-50, 450])
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def mean_std(data):
    return np.mean(data, axis=1), np.std(data, axis=1)


basic_IL_offline = NetworkGraph("2-agents/offline/IL_offline.py_06:20:03:43_adam-0.002.log")
basic_IL_offline.load()
basic_GL_offline = NetworkGraph("2-agents/offline/GL_offline.py_06:19:21:43_adam-0.002.log")
basic_GL_offline.load()
basic_PS_offline = NetworkGraph("2-agents/offline/PS_offline.py_06:20:00:33_adam-0.002.log")
basic_PS_offline.load()
basic_ML_offline = NetworkGraph("2-agents/offline/ML_offline.py_06:21:21:09_adam-0.002.log")
basic_ML_offline.load()
basic_MS_imitate = NetworkGraph("2-agents/offline/MS_imitate.py_06:20:08:34_adam-0.002.log")
basic_MS_imitate.load()

basic_IL_online = NetworkGraph("2-agents/online/IL_online.py_t_max1_06:19:11:25_adam-0.002.log")
basic_IL_online.load()
basic_GL_online = NetworkGraph("2-agents/online/GL_online.py_t_max_1_06:21:20:06_adam-0.002.log")
basic_GL_online.load()
basic_PS_online = NetworkGraph("2-agents/online/PS_online.py_t_max_1_06:22:09:40_adam-0.002.log")
basic_PS_online.load()
basic_ML_online = NetworkGraph("2-agents/online/ML_online.py_t_max_1_06:18:17:44_adam-0.002.log")
basic_ML_online.load()

toxin_IL_offline = NetworkGraph("2-agent-with-toxin/offline/IL_offline.py_06:20:06:04_adam-0.002.log")
toxin_IL_offline.load()
toxin_GL_offline = NetworkGraph("2-agent-with-toxin/offline/GL_offline.py_06:19:23:05_adam-0.002.log")
toxin_GL_offline.load()
toxin_PS_offline = NetworkGraph("2-agent-with-toxin/offline/PS_offline.py_06:20:02:05_adam-0.002.log")
toxin_PS_offline.load()
toxin_ML_offline = NetworkGraph("2-agent-with-toxin/offline/ML_offline.py_06_22_00_20_adam-0.002.log")
toxin_ML_offline.load()
toxin_MS_imitate = NetworkGraph("2-agent-with-toxin/offline/MS_imitate.py_06:20:10:21_adam-0.002.log")
toxin_MS_imitate.load()

toxin_IL_online = NetworkGraph("2-agent-with-toxin/online/IL_online.py_t_max1_06:19:12:53_adam-0.002.log")
toxin_IL_online.load()
toxin_GL_online = NetworkGraph("2-agent-with-toxin/online/GL_online.py_t_max_1_06:22:02:12_adam-0.002.log")
toxin_GL_online.load()
toxin_PS_online = NetworkGraph("2-agent-with-toxin/online/PS_online.py_t_max_1_06:22:11:09_adam-0.002.log")
toxin_PS_online.load()
toxin_ML_online = NetworkGraph("2-agent-with-toxin/online/ML_online.py_t_max_1_06:18:15:57_adam-0.002.log")
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


# def smooth(data):
#     a = len(data) % 48
#     return data[0:-a].reshape(-1, 48)


fig = plt.figure(1, figsize=(30, 20))

plt.subplot(221)
confidence_band_offline(smooth_offline(basic_IL_offline.reward), smooth_offline(basic_GL_offline.reward),
                        smooth_offline(basic_PS_offline.reward),
                        smooth_offline(basic_ML_offline.reward),
                        smooth_offline(basic_MS_imitate.reward),
                        )
ax = plt.gca()
ax.set_title("Offline updating methods in 2-10-0")
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Running epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

plt.subplot(222)
confidence_band_offline(smooth_offline(toxin_IL_offline.reward), smooth_offline(toxin_GL_offline.reward),
                        smooth_offline(toxin_PS_offline.reward),
                        smooth_offline(toxin_ML_offline.reward),
                        smooth_offline(toxin_MS_imitate.reward),
                        )
ax = plt.gca()
ax.set_title("Offline updating methods in 2-10-5")
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Running epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

plt.subplot(223)
confidence_band_online(smooth_online(basic_IL_online.rewards), smooth_online(basic_GL_online.rewards),
                       smooth_online(basic_PS_online.rewards),
                       smooth_online(basic_ML_online.rewards),
                       )
ax = plt.gca()
ax.set_title("Online updating methods in 2-10-0")
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Running epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

plt.subplot(224)
confidence_band_online(smooth_online(toxin_IL_online.rewards), smooth_online(toxin_GL_online.rewards),
                       smooth_online(toxin_PS_online.rewards),
                       smooth_online(toxin_ML_online.rewards),
                       )
ax = plt.gca()
ax.set_title("Online updating methods in 2-10-5")
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Running epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')
plt.tight_layout(pad=15, w_pad=10, h_pad=7)
plt.show()
