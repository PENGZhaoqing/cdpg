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

def range_x(data):
    return np.array(range(len(data))) * 0.5


def confidence_bar(data1, data2, data3, data4, data5):
    x1 = range_x(data1)
    x2 = range_x(data2)
    x3 = range_x(data3)
    x4 = range_x(data4)
    x5 = range_x(data5)
    # x6 = range_x(data6)
    # x7 = range_x(data7)

    mean1, std1 = mean_std(data1)
    mean2, std2 = mean_std(data2)
    mean3, std3 = mean_std(data3)
    mean4, std4 = mean_std(data4)
    mean5, std5 = mean_std(data5)
    # mean6, std6 = mean_std(data6)
    # mean7, std7 = mean_std(data7)

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


    # purple
    # plt.plot(x6, mean6, 'k', color='#a64dff', label="ML Bandwidth: 6")
    # plt.fill_between(x6, mean6 - std6, mean6 + std6,
    #                  alpha=0.1, edgecolor='#a64dff', facecolor='#a64dff',
    #                  linewidth=1)
    # # yellow
    # plt.plot(x7, mean7, 'k', color='#e6e600', label="ML Bandwidth: 8")
    # plt.fill_between(x7, mean7 - std7, mean7 + std7,
    #                  alpha=0.1, edgecolor='#e6e600', facecolor='#e6e600',
    #                  linewidth=1)

    plt.legend(loc=2, labelspacing=0)
    axes = plt.gca()
    axes.set_xlim([0, 40])
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def mean_std(data):
    return np.mean(data, axis=1), np.std(data, axis=1)


com_signal_1 = NetworkGraph("IL_offline.py_06:20:03:43_adam-0.002.log")
com_signal_1.load()
com_signal_2 = NetworkGraph("GL_offline.py_06:19:21:43_adam-0.002.log")
com_signal_2.load()
com_signal_3 = NetworkGraph("PS_offline.py_06:20:00:33_adam-0.002.log")
com_signal_3.load()
com_signal_4 = NetworkGraph("ML_offline.py_06:21:21:09_adam-0.002.log")
com_signal_4.load()
com_signal_5 = NetworkGraph("MS_imitate.py_06:20:08:34_adam-0.002.log")
com_signal_5.load()

fig = plt.figure(1, figsize=(30, 6))
plt.subplot(122)


def smooth(data):
    a = len(data) % 2
    if a != 0:
        data = data[0:-a].reshape(-1, 2).mean(axis=1)
    else:
        data = data.reshape(-1, 2).mean(axis=1)
    data2 = data[0:80 * 24]
    return data2.reshape(-1, 24)
    # return data1.reshape(-1, 48)


# def smooth(data):
#     a = len(data) % 48
#     return data[0:-a].reshape(-1, 48)

confidence_bar(smooth(com_signal_1.reward), smooth(com_signal_2.reward), smooth(com_signal_3.reward),
               smooth(com_signal_4.reward),
               smooth(com_signal_5.reward),
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

plt.show()
