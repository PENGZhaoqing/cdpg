from io import open
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d
import numpy as np
import math
import os
from scipy import ndimage


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


def plot(x1, y1, x2, y2, x3, y3):
    plt.plot(x1, y1, 'b-', label="Full distributed method")
    plt.plot(x2, y2, 'r-', label="Cooperative distributed method (CDPG)")
    plt.plot(x3, y3, 'g-', label="Centralized method")
    plt.legend(loc=4, labelspacing=0)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')


def draw(mean_data1, mean_data2, mean_data3):
    x1 = range(len(mean_data1))
    x2 = range(len(mean_data2))
    x3 = range(len(mean_data3))
    plot(x1, mean_data1, x2, mean_data2, x3, mean_data3)


no_com_network = NetworkGraph("uncom_t_max3_06:07:22:06_adam-0.002.log")
no_com_network.load()
com_network = NetworkGraph("com_tmax_1_06:02:22:00_adam-0.002.log")
com_network.load()
cen_network = NetworkGraph("central_cdpg.py_06:07:17:11_adam-0.002.log")
cen_network.load()

fig = plt.figure(1, figsize=(30, 10))
plt.subplot(221)
draw(no_com_network.reward, com_network.reward, cen_network.reward)
plt.ylabel('scores')
plt.grid(True)

plt.subplot(222)
draw(no_com_network.policies, com_network.policies, cen_network.policies)
plt.ylabel('polices')
axes = plt.gca()
axes.set_ylim([0, 1.2])
plt.grid(True)

plt.subplot(223)
draw(no_com_network.avg_adv, com_network.avg_adv, cen_network.avg_adv)
axes = plt.gca()
axes.set_ylim([-0.2,0.1])
plt.ylabel('avg adv')

plt.grid(True)
plt.subplot(224)
draw(no_com_network.avg_Q_value, com_network.avg_Q_value, cen_network.avg_Q_value)
plt.ylabel('avg values')
axes = plt.gca()
axes.set_ylim([-5,30])
plt.grid(True)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')

plt.show()
