import matplotlib.pyplot as plt
import numpy as np


# Function to plot normalized confusion matrix
def plot_norm_confusion(confusion, title, cmap='viridis'):
    # Normalize the confusion by row
    norm_confusion = confusion / (np.sum(confusion, axis=1).reshape(len(confusion), 1))
    norm_confusion = np.flip(norm_confusion, axis=0)  # Flip y axis
    norm_confusion = np.flip(norm_confusion, axis=1)  # Flip x axis

    fig, ax = plt.subplots()
    cax = ax.imshow(norm_confusion, cmap=cmap)

    # Add a color bar
    plt.colorbar(cax)

    ticks = ['N3', 'N1/N2', 'REM', 'W']
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(ticks)), ticks[::-1])
    ax.set_yticks(np.arange(len(ticks)), ticks[::-1])

    # Loop over data dimensions and create text annotations.
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            color = 'white' if norm_confusion[i, j] < 0.5 else 'black'
            text = ax.text(j, i, np.round(norm_confusion[i, j], decimals=2),
                           ha="center", va="center", color=color)

    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    confusion = np.array([[10844, 25921,   778,  2722],
 [10006, 84583,  7725, 17311],
 [  245, 11014, 27828,  2792],
 [  583,  8381,  1302, 55663]])
    plot_norm_confusion(confusion, "", cmap="plasma")
