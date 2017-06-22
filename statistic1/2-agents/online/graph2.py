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


def plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7):
    plt.plot(x1, y1, 'b-', label="message length:1")
    plt.plot(x2, y2, 'r-', label="message length:2")
    plt.plot(x3, y3, 'g-', label="message length:3")
    plt.plot(x4, y4, 'y-', label="message length:4")
    plt.plot(x5, y5, 'm-', label="message length:5")
    plt.plot(x6, y6, 'c-', label="message length:6")
    plt.plot(x7, y7, '0.5', label="message length:8")
    plt.legend(loc=4, labelspacing=0)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def draw(mean_data1, mean_data2, mean_data3, mean_data4, mean_data5, mean_data6, mean_data7):
    x1 = range(len(mean_data1))
    x2 = range(len(mean_data2))
    x3 = range(len(mean_data3))
    x4 = range(len(mean_data4))
    x5 = range(len(mean_data5))
    x6 = range(len(mean_data6))
    x7 = range(len(mean_data7))
    plot(x1, mean_data1, x2, mean_data2, x3, mean_data3, x4, mean_data4, x5, mean_data5, x6, mean_data6, x7, mean_data7)


def range_x(data):
    return np.array(range(len(data))) * 0.5


def confidence_bar(data1, data2, data3, data4):
    x1 = range_x(data1)
    x2 = range_x(data2)
    x3 = range_x(data3)
    x4 = range_x(data4)

    mean1, std1 = mean_std(data1)
    mean2, std2 = mean_std(data2)
    mean3, std3 = mean_std(data3)
    mean4, std4 = mean_std(data4)

    # # red
    plt.plot(x1, mean1, 'k', color='#c46ea0', label="GL online")
    plt.fill_between(x1, mean1 - std1, mean1 + std1,
                     alpha=0.1, edgecolor='#c46ea0', facecolor='#c46ea0',
                     linewidth=1)
    # # greeen
    plt.plot(x2, mean2, 'k', color='#009E73', label="IL online")
    plt.fill_between(x2, mean2 - std2, mean2 + std2,
                     alpha=0.1, edgecolor='#009E73', facecolor='#009E73',
                     linewidth=1)
    # # red
    plt.plot(x3, mean3, 'k', color='#ff0066', label="ML online")
    plt.fill_between(x3, mean3 - std3, mean3 + std3,
                     alpha=0.1, edgecolor='#ff0066', facecolor='#ff0066',
                     linewidth=1)

    # # orange
    plt.plot(x4, mean4, 'k', color='#ff9900', label="PS online")
    plt.fill_between(x4, mean4 - std4, mean4 + std4,
                     alpha=0.1, edgecolor='#ff9900', facecolor='#ff9900',
                     linewidth=1)

    plt.legend(loc=4, labelspacing=0)
    axes = plt.gca()
    axes.set_xlim([0, 40])
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def mean_std(data):
    return np.mean(data, axis=1), np.std(data, axis=1)


com_signal_1 = NetworkGraph("GL_online.py_t_max_1_06:21:21:14_adam-0.002.log")
com_signal_1.load()
com_signal_2 = NetworkGraph("IL_online.py_t_max1_06:19:11:25_adam-0.002.log")
com_signal_2.load()
com_signal_3 = NetworkGraph("ML_online.py_t_max_1_06:18:17:44_adam-0.002.log")
com_signal_3.load()
com_signal_4 = NetworkGraph("PS_online.py_t_max_1_06:22:09:40_adam-0.002.log")
com_signal_4.load()

fig = plt.figure(1, figsize=(30, 6))
plt.subplot(122)


def smooth(data):
    return data.reshape(80, -1, 32).mean(axis=1)


confidence_bar(smooth(com_signal_1.rewards), smooth(com_signal_2.rewards), smooth(com_signal_3.rewards),
               smooth(com_signal_4.rewards),
               )
ax = plt.gca()
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Epoch')
plt.grid(color='w', linestyle='-', linewidth=1)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')

plt.show()
