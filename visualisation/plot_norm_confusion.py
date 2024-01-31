import matplotlib.pyplot as plt
import numpy as np


# Function to plot confusion matrix
def plot_norm_confusion(confusion, title):
    # Normalise the confusion by row
    print(confusion)
    norm_confusion = confusion / (np.sum(confusion, axis=1).reshape(len(confusion), 1))
    norm_confusion = np.flip(norm_confusion, axis=0)    # Lionel prefers the confusion matrix figure to have a flipped y axis.

    fig, ax = plt.subplots()
    plt.imshow(norm_confusion)

    ticks = ['N3','N1/N2','REM','W']

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(ticks)), ticks)
    ax.set_yticks(np.arange(len(ticks)), ticks[::-1])  # Lionel prefers flipped y axis.

    # Loop over data dimensions and create text annotations.
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            text = ax.text(j, i, np.round(norm_confusion[i, j], decimals=2),
                           ha="center", va="center", color=str(1-norm_confusion[i, j]))

    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    confusion = np.array([[3.7921e+04, 9.2200e+02 ,6.2000e+01, 1.4900e+02],
 [1.6902e+04, 8.5477e+04, 6.7440e+03, 1.4020e+04],
 [2.0100e+02, 1.2160e+03, 3.5449e+04, 5.8550e+03],
 [9.1600e+02, 1.1460e+03, 7.2800e+02, 7.3304e+04]])
    plot_norm_confusion(confusion, "")
