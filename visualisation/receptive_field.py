import matplotlib.pyplot as plt
import numpy as np

"""
This script takes a series of models represented by a list of dict. It then plots the receptive field vs. layer number.

model = {'k':    [],   # Kernel sizes for each layer
          's':    [],   # The downsampling factor at each layer due to previous strided convolution/downsampling.
          'f':    125,
          'd':    3750}

Note: if there is global max pooling at the end, make the last s a large number, to ensure that the receptive field maxes out at d.
"""

experiment1 = {'k': [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3],
                   's': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024],
                   'f': 125,
                   'd': 3750,
                   'name': 'Experiment 1 (2nd Best)',
                   'color': '#0077B6'}

experiment2 = {'k': [21, 21, 21, 21, 21, 21, 21],
                     's': [1, 2, 4, 8, 16, 32, 64],
                     'f': 125,
                     'd': 3750,
                     'name': 'Experiment 2 (Worst)',
                     'color': '#90E0Ef'}

experiment3 = {'k': [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3],
                   's': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024],
                   'f': 31.25,
                   'd': 3748,
                   'name': 'Experiment 3 (Best)',
                   'color': '#03045E'}

experiment4 = {'k': [21, 21, 21, 21, 21, 21, 21],
                     's': [1, 2, 4, 8, 16, 32, 9999999999999999],
                     'f': 31.25,
                     'd': 3748,
                     'name': 'Experiment 4 (3rd Best)',
                     'color': '#00B4D8'}

experiment5 = {'k': [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3],
                   's': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024],
                   'f': 125,
                   'd': 3750,
                   'name': 'Experiment 5 (2nd Best)',
                   'color': '#B30000'}

experiment6 = {'k': [21, 21, 21, 21, 21, 21, 21],
                     's': [1, 2, 4, 8, 16, 32, 64],
                     'f': 125,
                     'd': 3750,
                     'name': 'Experiment 6 (3rd Best)',
                     'color': '#FF2626'}

experiment7 = {'k': [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3],
                   's': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                   'f': 125,
                   'd': 15000,
                   'name': 'Experiment 7 (Best)',
                   'color': '#340002'}

experiment8 = {'k': [21, 21, 21, 21, 21, 21, 21],
                     's': [1, 2, 4, 8, 16, 32, 9999999999999999],
                     'f': 125,
                     'd': 15000,
                     'name': 'Experiment 8 (Worst)',
                     'color': '#EE6B6E'}

rip_models = [experiment1, experiment2, experiment3, experiment4]
ecg_models = [experiment5, experiment6, experiment7, experiment8]
fig, (ax, ax1) = plt.subplots(1, 2)

for j, model in enumerate(rip_models):
    k, s, f, d = model['k'], model['s'], model['f'], model['d']
    # Check correct definition of model layers.
    assert len(k) == len(s)
    n_layers = len(k)
    # Find receptive field of each layer's output
    r=[1, k[0]] # Receptive field of input and first layer
    for i in range(1, n_layers):
        r.append(min(r[i-1] + (k[i]-1) * s[i], d))
        # Once receptive field = input dimensionality it maxes out.
    r = np.array(r) / f  # Convert from samples to seconds

    #Plot
    ax.plot(np.arange(n_layers+1),r, label=model['name'], color=model['color'])

for j, model in enumerate(ecg_models):
    k, s, f, d = model['k'], model['s'], model['f'], model['d']
    # Check correct definition of model layers.
    assert len(k) == len(s)
    n_layers = len(k)
    # Find receptive field of each layer's output
    r=[1, k[0]] # Receptive field of input and first layer
    for i in range(1, n_layers):
        r.append(min(r[i-1] + (k[i]-1) * s[i], d))
        # Once receptive field = input dimensionality it maxes out.
    r = np.array(r) / f  # Convert from samples to seconds

    #Plot
    ax1.plot(np.arange(n_layers+1),r, label=model['name'], color=model['color'])



ax.set_ylabel('Receptive field (s)', fontsize=12)
#ax.set_xlabel('Layer no.')
ax.set_yscale('linear')
ax.legend()
ax.set_title('RIP')

ax1.legend()
ax1.set_title('ECG')
fig.text(0.5, 0.04, 'Layer no.', ha='center', fontsize=12)

plt.show()

