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


def plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5):
    plt.plot(x1, y1, 'b-', label="message length:1")
    plt.plot(x2, y2, 'r-', label="message length:2")
    plt.plot(x3, y3, 'g-', label="message length:4")
    plt.plot(x4, y4, 'y-', label="message length:8")
    plt.plot(x5, y5, 'k-', label="message length:16")

    plt.legend(loc=4, labelspacing=0)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')



def draw(mean_data1, mean_data2, mean_data3, mean_data4, mean_data5):
    x1 = range(len(mean_data1))
    x2 = range(len(mean_data2))
    x3 = range(len(mean_data3))
    x4 = range(len(mean_data4))
    x5 = range(len(mean_data5))

    plot(x1, mean_data1, x2, mean_data2, x3, mean_data3, x4, mean_data4, x5, mean_data5)

com_signal_1 = NetworkGraph("signal_1_cdpg.py_06:06:21:54_adam-0.002.log")
com_signal_1.load()
com_signal_2 = NetworkGraph("signal_2_com_tmax_1_06:03:16:33_adam-0.002.log")
com_signal_2.load()
com_signal_4 = NetworkGraph("com_tmax_1_06:02:22:00_adam-0.002.log")
com_signal_4.load()
com_signal_8 = NetworkGraph("signal_8_com_tmax_1_06:03:18:26_adam-0.002.log")
com_signal_8.load()
com_signal_16 = NetworkGraph("signal_16_cdpg.py_06:06:19:51_adam-0.002.log")
com_signal_16.load()

fig = plt.figure(1, figsize=(30, 6))
plt.subplot(121)
draw(com_signal_1.reward, com_signal_2.reward, com_signal_4.reward, com_signal_8.reward, com_signal_16.reward)
plt.ylabel('scores')
plt.grid(True)


# plt.grid(True)
# plt.subplot(122)
# draw(com_signal_4.avg_Q_value, com_signal_8.avg_Q_value, com_signal_16.avg_Q_value)
# plt.ylabel('avg values')
# axes = plt.gca()
# axes.set_ylim([-5,30])
# plt.grid(True)

# fig.tight_layout()
# plt.setp(lines, color='b', linewidth=1.0)
# plt.title(r'$\sigma_i=15$')

plt.show()
