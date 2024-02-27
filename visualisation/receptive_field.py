import matplotlib.pyplot as plt
import numpy as np

"""
This script takes a series of models represented by a list of dict. It then plots the receptive field vs. layer number.

model = {'k':    [],   # Kernel sizes for each layer
          's':    [],   # The current downsampling factor due to previous strided convolution/downsampling.
          'f':    125,
          'd':    3750}
"""


sors_nocontext2 = {'k': [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3],
                   's': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024],
                   'f': 125,
                   'd': 3750,
                   'name': 'Sors_nocontext2',
                   'color': 'k'}

sors_largekernels = {'k': [21, 21, 21, 21, 21, 21, 21],
                     's': [1, 2, 4, 8, 16, 32, 64],
                     'f': 125,
                     'd': 3750,
                     'name': 'Sors_largekernels',
                     'color': 'b'}

models = [sors_nocontext2, sors_largekernels]
fig, ax = plt.subplots()

for j, model in enumerate(models):
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

ax.set_ylabel('Receptive field (s)')
ax.set_xlabel('Layer no.')
ax.set_yscale('linear')
ax.legend()
plt.show()

