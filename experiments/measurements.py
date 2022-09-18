#!/usr/bin/python3

# this script contains many hardcoded parts, not proud of it, was in a hurry

import time
import matplotlib.pyplot as plt
from python.launcher import run
from python.settings import *

algos = [
    'gauss', 'iforest', 'hstrees', 'loda'
]
datasets = featuresize_datasets
algo_args = featuresize_algo_args
values = featuresize_values
f1_scores = {algo: [] for algo in algos}
roc_scores = {algo: [] for algo in algos}
times = {algo: [] for algo in algos}
for dataset in datasets:
    for algo, algo_arg in zip(algos, algo_args):
        if 'num_feats' in algo_arg.keys():
            algo_arg['num_feats'] = dataset.shape[1] - 1
        print("Running with settings:")
        print("set: {}\nalgo: {}\nparams: {}".format(dataset, algo, algo_arg))
        print("Results:")
        start = time.process_time()
        f1, roc = run(dataset, algo, False, False, **algo_arg)
        finish = time.process_time()
        time_elapsed = finish - start
        f1_scores[algo].append(f1)
        roc_scores[algo].append(roc)
        times[algo].append(time_elapsed)
        print("CPU seconds elapsed: {}".format(time_elapsed))
        print()

plt.plot(values, times['gauss'], c='green', marker='o',
         label='gauss', linestyle=(0, ()))
plt.plot(values, times['iforest'], c='red', marker='^',
         label='iforest')
plt.plot(values, times['hstrees'], c='blue', marker='s',
         label='hstrees', linestyle=(0, (1, 1)))
plt.plot(values, times['loda'], c='black', marker='D',
         label='loda', linestyle=(0, (5, 5)))
plt.ylabel('Time of execution')
plt.xlabel('Number of examples in the dataset')
plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
           mode="expand", borderaxespad=0, ncol=4)
plt.show()

plt.subplot(2, 1, 1)
plt.plot(values, f1_scores['gauss'], c='green', marker='o',
         label='gauss', linestyle=(0, ()))
plt.plot(values, f1_scores['iforest'], c='red', marker='^',
         label='iforest')
plt.plot(values, f1_scores['hstrees'], c='blue', marker='s',
         label='hstrees', linestyle=(0, (1, 1)))
plt.plot(values, f1_scores['loda'], c='black', marker='D',
         label='loda', linestyle=(0, (5, 5)))
plt.ylabel('F1 Scores')
plt.legend(loc='upper left', mode='expand', ncol=4,
           bbox_to_anchor=(0, 1.02, 1, 0.2))

plt.subplot(2, 1, 2)
plt.plot(values, roc_scores['gauss'], c='green', marker='o',
         label='gauss', linestyle=(0, ()))
plt.plot(values, roc_scores['iforest'], c='red', marker='^',
         label='iforest')
plt.plot(values, roc_scores['hstrees'], c='blue', marker='s',
         label='hstrees', linestyle=(0, (1, 1)))
plt.plot(values, roc_scores['loda'], c='black', marker='D',
         label='loda', linestyle=(0, (5, 5)))
plt.ylabel('ROC AUC Scores')

plt.xlabel('Number of examples in the dataset')
plt.show()
