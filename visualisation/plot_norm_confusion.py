import matplotlib.pyplot as plt
import numpy as np


# Function to plot confusion matrix
def plot_norm_confusion(confusion, title):
    # Normalise the confusion by row
    print(confusion)
    norm_confusion = confusion / (np.sum(confusion, axis=1).reshape(len(confusion),1))

    fig, ax = plt.subplots()
    plt.imshow(norm_confusion)

    ticks = ['N3','N1/N2','REM','W']

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(np.arange(len(ticks)), ticks)
    ax.set_yticks(np.arange(len(ticks)), ticks)

    # Loop over data dimensions and create text annotations.
    for i in range(len(ticks)):
        for j in range(len(ticks)):
            text = ax.text(j, i, np.round(norm_confusion[i, j], decimals=2),
                           ha="center", va="center", color=str(1-norm_confusion[i, j]))

    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    confusion = np.array([[37921, 922, 62, 149],
                          [16902, 85477, 6744, 14020],
                          [201, 1216, 35449, 5855],
                          [916, 1146, 728, 73304]])
    plot_norm_confusion(confusion, "")
