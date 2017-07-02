from io import open
import matplotlib.pyplot as plt
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt


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


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


def range_x(data):
    return np.array(range(len(data))) * 0.5



def confidence_bar(data1, data2, data3, data4, data5, data6, data7):
    x1 = range_x(data1)
    x2 = range_x(data2)
    x3 = range_x(data3)
    x4 = range_x(data4)
    x5 = range_x(data5)
    x6 = range_x(data6)
    x7 = range_x(data7)

    mean1, std1 = mean_std(data1)
    mean2, std2 = mean_std(data2)
    mean3, std3 = mean_std(data3)
    mean4, std4 = mean_std(data4)
    mean5, std5 = mean_std(data5)
    mean6, std6 = mean_std(data6)
    mean7, std7 = mean_std(data7)

    # # red
    plt.plot(x1, mean1, 'k', color='#ff0066', label="CML Bandwidth: 1")
    plt.fill_between(x1, mean1 - std1, mean1 + std1,
                     alpha=0.1, edgecolor='#ff0066', facecolor='#ff0066',
                     linewidth=1)
    # # greeen
    plt.plot(x2, mean2, 'k', color='#009E73', label="CML Bandwidth: 2")
    plt.fill_between(x2, mean2 - std2, mean2 + std2,
                     alpha=0.1, edgecolor='#009E73', facecolor='#009E73',
                     linewidth=1)
    # # pink
    plt.plot(x3, mean3, 'k', color='#c46ea0', label="CML Bandwidth: 3")
    plt.fill_between(x3, mean3 - std3, mean3 + std3,
                     alpha=0.1, edgecolor='#c46ea0', facecolor='#c46ea0',
                     linewidth=1)

    # # orange
    plt.plot(x4, mean4, 'k', color='#ff9900', label="CML Bandwidth: 4")
    plt.fill_between(x4, mean4 - std4, mean4 + std4,
                     alpha=0.1, edgecolor='#ff9900', facecolor='#ff9900',
                     linewidth=1)

    # blue
    plt.plot(x5, mean5, 'k', color='#0B77B5', label="CML Bandwidth: 5")
    plt.fill_between(x5, mean5 - std5, mean5 + std5,
                     alpha=0.1, edgecolor='#0B77B5', facecolor='#0B77B5',
                     linewidth=1)
    # purple
    plt.plot(x6, mean6, 'k', color='#a64dff', label="CML Bandwidth: 6")
    plt.fill_between(x6, mean6 - std6, mean6 + std6,
                     alpha=0.1, edgecolor='#a64dff', facecolor='#a64dff',
                     linewidth=1)
    # yellow
    plt.plot(x7, mean7, 'k', color='#e6e600', label="CML Bandwidth: 8")
    plt.fill_between(x7, mean7 - std7, mean7 + std7,
                     alpha=0.1, edgecolor='#e6e600', facecolor='#e6e600',
                     linewidth=1)

    plt.legend(loc=4, labelspacing=0)
    axes = plt.gca()
    axes.set_xlim([0,40])
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def mean_std(data):
    return np.mean(data, axis=1), np.std(data, axis=1)


def smooth(data):
    return data.reshape(80, -1, 32).mean(axis=1)


toxin_signal_1 = NetworkGraph("1_cdpg.py_06_14_12_21_adam-0.002.log")
toxin_signal_1.load()
toxin_signal_2 = NetworkGraph("2_ML_online.py_t_max_1_06:18:21:59_adam-0.002.log")
toxin_signal_2.load()
toxin_signal_3 = NetworkGraph("3_cdpg.py_06_14_15_48_adam-0.002.log")
toxin_signal_3.load()
toxin_signal_4 = NetworkGraph("4_cdpg.py_06_14_17_35_adam-0.002.log")
toxin_signal_4.load()
toxin_signal_5 = NetworkGraph("5_cdpg.py_06_14_19_19_adam-0.002.log")
toxin_signal_5.load()
toxin_signal_6 = NetworkGraph("6_cdpg.py_06_14_21_04_adam-0.002.log")
toxin_signal_6.load()
toxin_signal_8 = NetworkGraph("8_ML_online.py_t_max_1_06:19:14:14_adam-0.002.log")
toxin_signal_8.load()

basic_signal_1 = NetworkGraph("../2-agent-signal/1_cdpg.py_06_13_00_49_adam-0.002.log")
basic_signal_1.load()
basic_signal_2 = NetworkGraph("../2-agent-signal/2_ML_online.py_t_max_1_06:18:19:48_adam-0.002.log")
basic_signal_2.load()
basic_signal_3 = NetworkGraph("../2-agent-signal/3_ML_online.py_t_max_1_06:19:10:33_adam-0.002.log")
basic_signal_3.load()
basic_signal_4 = NetworkGraph("../2-agent-signal/4_cdpg.py_06_13_05_31_adam-0.002.log")
basic_signal_4.load()
basic_signal_5 = NetworkGraph("../2-agent-signal/5_ML_online.py_t_max_1_06:18:18:02_adam-0.002.log")
basic_signal_5.load()
basic_signal_6 = NetworkGraph("../2-agent-signal/6_cdpg.py_06_13_07_07_adam-0.002.log")
basic_signal_6.load()
basic_signal_8 = NetworkGraph("../2-agent-signal/8_ML_online.py_t_max_1_06:19:12:29_adam-0.002.log")
basic_signal_8.load()

fig = plt.figure(1, figsize=(30, 5))
ax = plt.subplot(131)
ax.set_title("Running performance without toxin")
confidence_bar(smooth(basic_signal_1.rewards), smooth(basic_signal_2.rewards), smooth(basic_signal_3.rewards),
               smooth(basic_signal_4.rewards),
               smooth(basic_signal_5.rewards),
               smooth(basic_signal_6.rewards),
               smooth(basic_signal_8.rewards))
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Training epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

ax = plt.subplot(132)
ax.set_title("Running performance with 5 toxins")
confidence_bar(smooth(toxin_signal_1.rewards), smooth(toxin_signal_2.rewards), smooth(toxin_signal_3.rewards),
               smooth(toxin_signal_4.rewards),
               smooth(toxin_signal_5.rewards),
               smooth(toxin_signal_6.rewards),
               smooth(toxin_signal_8.rewards))
ax.set_facecolor('#EAEAF2')
plt.ylabel('Scores')
plt.xlabel('Training epochs')
plt.grid(color='w', linestyle='-', linewidth=1)

ax = plt.subplot(133)
N = 7
ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars
toxin_means = (np.mean(toxin_signal_1.rewards), np.mean(toxin_signal_2.rewards),
               np.mean(toxin_signal_3.rewards), np.mean(toxin_signal_4.rewards),
               np.mean(toxin_signal_5.rewards), np.mean(toxin_signal_6.rewards),
               np.mean(toxin_signal_8.rewards))
toxin_std = (np.std(toxin_signal_1.rewards), np.std(toxin_signal_2.rewards),
             np.std(toxin_signal_3.rewards), np.std(toxin_signal_4.rewards),
             np.std(toxin_signal_5.rewards), np.std(toxin_signal_6.rewards),
             np.std(toxin_signal_8.rewards))

rects1 = ax.bar(ind, toxin_means, width, color='r', yerr=toxin_std, ecolor='0.75', capsize=4)

basic_means = (np.mean(basic_signal_1.rewards), np.mean(basic_signal_2.rewards),
               np.mean(basic_signal_3.rewards), np.mean(basic_signal_4.rewards),
               np.mean(basic_signal_5.rewards), np.mean(basic_signal_6.rewards),
               np.mean(basic_signal_8.rewards))
basic_stds = (np.std(basic_signal_1.rewards), np.std(basic_signal_2.rewards),
              np.std(basic_signal_3.rewards), np.std(basic_signal_4.rewards),
              np.std(basic_signal_5.rewards), np.std(basic_signal_6.rewards),
              np.std(basic_signal_8.rewards))

rects2 = ax.bar(ind + width, basic_means, width, color='c', yerr=basic_stds, ecolor='0.75', capsize=4)
ax.set_title("Average performance on diverse bandwidths")
ax.set_ylabel('Average scores')
ax.set_xlabel('CML Bandwidth')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '8'))
ax.legend((rects1[0], rects2[0]), ('5 toxins', 'no toxin'), loc=2)
autolabel(rects1)
autolabel(rects2)
plt.tight_layout(pad=4, w_pad=5, h_pad=0)
plt.show()
