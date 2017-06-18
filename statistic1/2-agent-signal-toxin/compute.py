from io import open
import matplotlib.pyplot as plt
import numpy as np


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


com_signal_1 = NetworkGraph("1_cdpg.py_06:14:12:21_adam-0.002.log")
com_signal_1.load()
com_signal_2 = NetworkGraph("2_cdpg.py_06:14:14:05_adam-0.002.log")
com_signal_2.load()
com_signal_3 = NetworkGraph("3_cdpg.py_06:14:15:48_adam-0.002.log")
com_signal_3.load()
com_signal_4 = NetworkGraph("4_cdpg.py_06:14:17:35_adam-0.002.log")
com_signal_4.load()
com_signal_5 = NetworkGraph("5_cdpg.py_06:14:19:19_adam-0.002.log")
com_signal_5.load()
com_signal_6 = NetworkGraph("6_cdpg.py_06:14:21:04_adam-0.002.log")
com_signal_6.load()
com_signal_8 = NetworkGraph("8_cdpg.py_06:14:22:48_adam-0.002.log")
com_signal_8.load()

basic_signal_1 = NetworkGraph("../2-agent-signal/1_cdpg.py_06:13:00:49_adam-0.002.log")
basic_signal_1.load()
basic_signal_2 = NetworkGraph("../2-agent-signal/2_cdpg.py_06:13:02:24_adam-0.002.log")
basic_signal_2.load()
basic_signal_3 = NetworkGraph("../2-agent-signal/3_cdpg.py_06:13:03:57_adam-0.002.log")
basic_signal_3.load()
basic_signal_4 = NetworkGraph("../2-agent-signal/4_cdpg.py_06:13:05:31_adam-0.002.log")
basic_signal_4.load()
basic_signal_5 = NetworkGraph("../2-agent-signal/6_cdpg.py_06:13:07:07_adam-0.002.log")
basic_signal_5.load()
basic_signal_6 = NetworkGraph("../2-agent-signal/6_cdpg.py_06:13:07:07_adam-0.002.log")
basic_signal_6.load()
basic_signal_8 = NetworkGraph("../2-agent-signal/8_cdpg.py_06:13:08:42_adam-0.002.log")
basic_signal_8.load()

import numpy as np
import matplotlib.pyplot as plt

N = 7
toxin_means = (np.mean(com_signal_1.reward), np.mean(com_signal_2.reward),
               np.mean(com_signal_3.reward), np.mean(com_signal_4.reward),
               np.mean(com_signal_5.reward), np.mean(com_signal_6.reward),
               np.mean(com_signal_8.reward))
toxin_std = (np.std(com_signal_1.reward), np.std(com_signal_2.reward),
             np.std(com_signal_3.reward), np.std(com_signal_4.reward),
             np.std(com_signal_5.reward), np.std(com_signal_6.reward),
             np.std(com_signal_8.reward))

ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, toxin_means, width, color='r', yerr=toxin_std, ecolor='0.75')

basic_means = (np.mean(basic_signal_1.reward), np.mean(basic_signal_2.reward),
               np.mean(basic_signal_3.reward), np.mean(basic_signal_4.reward),
               np.mean(basic_signal_5.reward), np.mean(basic_signal_6.reward),
               np.mean(basic_signal_8.reward))
basic_stds = (np.std(basic_signal_1.reward), np.std(basic_signal_2.reward),
              np.std(basic_signal_3.reward), np.std(basic_signal_4.reward),
              np.std(basic_signal_5.reward), np.std(basic_signal_6.reward),
              np.std(basic_signal_8.reward))

rects2 = ax.bar(ind + width, basic_means, width, color='c', yerr=basic_stds, ecolor='0.75')

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5', 'Signal 6', 'Signal 8'))

ax.legend((rects1[0], rects2[0]), ('Toxin', 'Basic'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

# import matplotlib.pyplot as plt
# import numpy as np
#
# # construct some data like what you have:
# x = np.random.randn(100, 8)
# mins = x.min(0)
# maxes = x.max(0)
# means = x.mean(0)
# std = x.std(0)
#
# # create stacked errorbars:
# plt.errorbar(np.arange(8), means, std, fmt='ok', lw=3)
# plt.errorbar(np.arange(8), means, [means - mins, maxes - means],
#              fmt='.k', ecolor='gray', lw=1)
# plt.xlim(-1, 8)

plt.show()
