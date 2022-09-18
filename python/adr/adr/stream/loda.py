"""
Author: Nick Gavalas <gavalnick@gmail.com>

Implementation of Lightweight Online Detector of Anomalies algorithm based on
Penvý's paper https://link.springer.com/content/pdf/10.1007%2Fs10994-015-5521-0.pdf)
"""

# NOTE TODO: NOT TESTED

import numpy as np


class Histogram:

    def __init__(self, ndim, bwidth=1, random_state=None):
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random

        assert ndim > 0
        self.ndim = ndim
        assert bwidth > 0
        self.bwidth = bwidth

        self.hist = dict()
        self.pvec = self._make_pvec()
        # map each point to its bucket
        self.points = dict()

    def _make_pvec(self):
        vector = np.zeros(self.ndim)
        indices = self.rng.choice(self.ndim, int(np.floor(np.sqrt(self.ndim))), replace=False)
        for index in indices:
            vector[index] = self.rng.normal(0, 1)
        return vector

    def insert(self, point, index):
        assert point.shape[0] == 1
        assert point.shape[1] == self.ndim
        assert index not in self.points

        dot = np.dot(self.pvec, point[0])
        bucket = int(dot // self.bwidth)

        if bucket not in self.hist:
            self.hist[bucket] = 0
        self.hist[bucket] += 1
        self.points[index] = bucket

    def remove(self, index):
        assert index in self.points

        bucket = self.points[index]
        self.hist[bucket] -= 1
        del self.points[index]

    def score(self, index):
        assert index in self.points

        bucket = self.points[index]
        prob = self.hist[bucket]
        assert prob > 0
        return np.log(prob)


class LODA:
    """
    Lightweight Online Detector of Anomalies

    Arguments:
        ndim {int} -- Dimensionality

    Keyword Arguments:
        bwidth {int} -- Bucket width (default: {1})
        nvec: {int} -- Number of projection vectors (default: {10})
        random_state: {numpy.random.RandomState|int} -- Random state. If
            not specified, np.random is used. (default: {None})
    """

    def __init__(self, ndim, bwidth=1, nvec=10, random_state=None):
        self.ndim = ndim
        self.bwidth = bwidth
        self.nvec = nvec

        self._histograms = [Histogram(ndim, bwidth=bwidth, random_state=random_state) for _ in range(self.nvec)]

    def insert(self, point, index):
        for histogram in self._histograms:
            histogram.insert(point, index)

    def remove(self, index):
        for histogram in self._histograms:
            histogram.remove(index)

    def score(self, index):
        score = 0
        for histogram in self._histograms:
            score += histogram.score(index)
        score *= -1.0 / self.nvec
        return score
