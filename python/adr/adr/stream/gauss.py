"""
Author: Nick Gavalas <gavalnick@gmail.com>

Implementation of the Multivariate Gaussian Model for Anomaly Detection based
on https://www.coursera.org/learn/machine-learning/resources/szFCa.

See also: https://www.coursera.org/learn/machine-learning/resources/szFCa
and http://cs229.stanford.edu/section/gaussians.pdf
"""

import numpy as np


class MultivariateGaussian:

    def __init__(self, ndim):
        self.ndim = ndim

        self.sums = np.zeros(ndim)
        self.squares = np.zeros(ndim)
        self.count = 0

        self.points = {}
    
    def insert(self, point, index):
        assert point.shape[0] == 1
        assert point.shape[1] == self.ndim
        assert index not in self.points

        self.sums += point[0]
        self.squares += np.power(point[0], 2)
        self.count += 1

        self.points[index] = point

    def remove(self, index):
        assert index in self.points

        point = self.points.pop(index)

        self.sums -= point[0]
        self.squares -= np.power(point[0], 2)
        self.count -= 1

    def _pdf(self, point):
        mu = self.sums / self.count
        sigma2 = (self.squares / self.count) - np.power(mu, 2)
        # HACK to avoid division by zero
        if (sigma2 == 0).any():
            return np.zeros(point.shape[1])
        return np.exp(- np.power(point - mu, 2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

    def score(self, index):
        assert index in self.points

        return np.prod(self._pdf(self.points[index]))
