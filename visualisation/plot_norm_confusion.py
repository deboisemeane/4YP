import matplotlib.pyplot as plt
import numpy as np


# Function to plot confusion matrix
def plot_norm_confusion(confusion, title):
    # Normalise the confusion by row
    print(confusion)
    norm_confusion = confusion / (np.sum(confusion, axis=1).reshape(len(confusion), 1))
    norm_confusion = np.flip(norm_confusion, axis=0)    # Lionel prefers the confusion matrix figure to have a flipped y axis.
    norm_confusion = np.flip(norm_confusion, axis=1)    # Lionel prefers the confusion matrix to have flipped x axis.
    fig, ax = plt.subplots()
    plt.imshow(norm_confusion)

    ticks = ['N3','N1/N2','REM','W']

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(ticks)), ticks[::-1])  # Lionel prefers flipped x axis
    ax.set_yticks(np.arange(len(ticks)), ticks[::-1])  # Lionel prefers flipped y axis.

    # Loop over data dimensions and create text annotations.
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            text = ax.text(j, i, np.round(norm_confusion[i, j], decimals=2),
                           ha="center", va="center", color=str(1-norm_confusion[i, j]))

    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    confusion = np.array([[3.8220e+04, 1.8600e+03, 2.0000e+00, 2.6500e+02],
 [1.4048e+04, 9.7393e+04, 3.4140e+03, 7.4130e+03],
 [1.6800e+02, 3.3980e+03, 3.4827e+04, 5.0320e+03],
 [4.8700e+02, 2.2990e+03, 2.9300e+02, 7.3662e+04]])
    plot_norm_confusion(confusion, "")
