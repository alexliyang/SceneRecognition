"""
Instruction:

Pass this program the name of saved stats and it will plot accuracy and training error.
EX: python plot_weights nn_stats.npz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def DisplayPlot(train, ylabel, title, number=0):
    """Displays training curve.

    Args:
        train: Training statistics.
        valid: Validation statistics.
        ylabel: Y-axis label of the plot.
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    plt.plot(train[:, 0], train[:, 1])
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.draw()
    plt.show()

if __name__ == '__main__':
    arguments = sys.argv[1:]
    stats = np.load(arguments[0])
    train_acc_list = stats['train_acc']
    train_ce_list = stats['train_ce']
    DisplayPlot(train_ce_list, 'Cross Entropy', 'CNN Error vs. Epoch')
    DisplayPlot(train_acc_list, 'Accuracy','CNN Accuracy vs. Epoch', number=1)
