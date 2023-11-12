import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Function to plot median train and val losses, and plus minus 1 IQR
# e.g. when you have trained multiple times with different splits.
# Arguments: a list of training loss arrays and a list of validation loss arrays.


def plot_losses(val_losses, train_losses, title, averaging="median", plus_minus=1, t_color="mediumblue", v_color="gold",
                fill_alpha=0.2):
    fig, ax = plt.subplots()

    vls = np.array(val_losses)
    tls = np.array(train_losses)

    if averaging == "median":
        # Calculate median loss at each training epoch, across the different splits.
        av_vls = np.median(vls, axis=0)
        av_tls = np.median(tls, axis=0)
        # Calculate the interquartile range at each epoch
        dev_vls = stats.iqr(vls, axis=0)
        dev_tls = stats.iqr(tls, axis=0)
        # Produce plots
        upper_vls = av_vls + (dev_vls * plus_minus)
        lower_vls = av_vls - (dev_vls * plus_minus)
        upper_tls = av_tls + (dev_tls * plus_minus)
        lower_tls = av_tls - (dev_tls * plus_minus)
        x = np.arange(len(av_vls))

        ax.plot(av_vls, label=(averaging + " validation loss"), color=v_color)
        ax.plot(upper_vls, color=v_color, linestyle='--',
                label=(averaging + r" validation loss $\pm$  "+str(plus_minus)+" IQR"),)
        ax.plot(lower_vls, color=v_color, linestyle='--')
        ax.fill_between(x, upper_vls, lower_vls, color=v_color, alpha=fill_alpha)

        ax.plot(av_tls, label=(averaging + " training loss"), color="mediumblue")
        ax.plot(upper_tls, color=t_color, linestyle='--',
                label=(averaging + r" training loss $\pm$  "+str(plus_minus)+" IQR"),)
        ax.plot(lower_tls, color=t_color, linestyle='--')
        ax.fill_between(x, upper_tls, lower_tls, color=t_color, alpha=fill_alpha)

    elif averaging == "mean":
        # Calculate mean loss at each training epoch, across the different splits.
        av_vls = np.mean(vls, axis=0)
        av_tls = np.mean(tls, axis=0)
        # Calculate the standard deviation at each epoch.
        dev_vls = np.std(vls, axis=0)
        dev_tls = np.std(vls, axis=0)
        raise ValueError("Unsupported averaging type.")
    else:
        raise ValueError("Unsupported averaging type.")

    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.show()
