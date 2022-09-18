import numpy as np
import matplotlib.pyplot as plt

from adr.util import rolling_window

from adr.stream.gauss import MultivariateGaussian
from adr.stream.hstrees import HSF
from adr.stream.loda import LODA
from adr.stream.rrcf import RRCF, IF

# settings
COLORS = {
    'GAUSS': 'purple',
    'HSF': 'orange',
    'LODA': 'black',
    'RRCF': 'red',
    'IF': 'green'
}

N = 730
NUM_TREES = 5
WINDOW_SIZE = 400
TRAIN_SIZE = 256

SEED = 12
ADD_NOISE = True

# Change accordingly
SELECTED_ALGOS = {
#    'GAUSS',
#    'HSF',
#    'LODA',
#    'RRCF',
    'IF'
}
PLOT_SIGNAL = False #True

rng = np.random.RandomState(SEED)

# Dataset with random uniform noise
t = np.arange(N)
sin = np.zeros((N, 1))
for i in range(N):
    noise = rng.uniform(0, 1) if ADD_NOISE else 0
    sin[i] = 50 * np.sin((2 * np.pi / 100) * (i - 30)) + 100 + noise
# Inject anomaly
for i in range(235, 255):
    noise = rng.uniform(0, 1) if ADD_NOISE else 0
    sin[i] = 80 + noise

sin = sin.reshape(len(sin), 1)

models = {}
if 'GAUSS' in SELECTED_ALGOS:
    models['GAUSS'] = MultivariateGaussian(WINDOW_SIZE)
if 'HSF' in SELECTED_ALGOS:
    models['HSF'] = HSF(np.array([[50, 50, 50, 50]]), np.array([[150, 150, 150, 150]]), random_state=rng)
if 'LODA' in SELECTED_ALGOS:
    models['LODA'] = LODA(WINDOW_SIZE, bwidth=1, nvec=50, random_state=rng)
if 'RRCF' in SELECTED_ALGOS:
    models['RRCF'] = RRCF(WINDOW_SIZE, NUM_TREES, random_state=rng)
if 'IF' in SELECTED_ALGOS:
    models['IF'] = IF(WINDOW_SIZE, NUM_TREES, random_state=rng)

res = {}
for name in models.keys():
    res[name] = np.zeros(N, dtype=np.float64)

for index, point in enumerate(rolling_window(sin, size=WINDOW_SIZE)):
    if index > TRAIN_SIZE:
        for model in models.values():
            model.remove(index - TRAIN_SIZE)

    for model in models.values():
        model.insert(point.reshape(1, WINDOW_SIZE), index)

    for name, vector in res.items():
        if name == 'IF':
            vector[index] += models[name].score(index, WINDOW_SIZE)
        else:
            vector[index] += models[name].score(index)

fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_SIGNAL:
    ax.plot(t, sin, c='blue')
for name, vector in res.items():
    ax.plot(t, vector, c=COLORS[name])
plt.show()
