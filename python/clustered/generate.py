#!/usr/bin/env python3

"""
Tool for generating synthetic datasets for tests and model evaluation.
'inliers' are generated from the Gaussian Distribution.
'outliers' are drawn from the Uniform Distribution.
"""

from sys import stdout

import numpy as np
from sklearn.datasets import make_blobs

from adr.util import normalize

# sklearn's default standard deviation for the gaussian blobs
STD = 1.0


def euclidean_distance(array, point):
    return np.sqrt(np.sum(np.square(array - point), axis=1))


def make_dataset(
        n_inliers=950,
        n_outliers=50,
        feats=2,
        blobs=1,
        seed=11,
        outrange=(-10, 10),
        unlabeled=False,
        dirty=False,
        randomized=False,
        enumerated=False,
        normalized=False,
        outfile=stdout
        ):

    rng = np.random.RandomState(seed=seed)

    inliers, _, centers = make_blobs(
        n_samples=n_inliers,
        n_features=feats,
        random_state=rng,
        centers=blobs,
        center_box=outrange,
        return_centers=True
    )

    if not unlabeled:
        inliers = np.concatenate([inliers, np.zeros((n_inliers, 1))], axis=1)

    if n_outliers > 0:
        outliers = rng.uniform(outrange[0], outrange[1], size=(n_outliers, feats))
        if not dirty:
            for center in centers:
                while True:
                    misplaced_inliers_indices = euclidean_distance(outliers, center) < STD
                    num_misplaced = np.sum(misplaced_inliers_indices)
                    if num_misplaced == 0:
                        break
                    outliers[misplaced_inliers_indices] = rng.uniform(outrange[0], outrange[1],
                                                                      size=(num_misplaced, feats))
        if not unlabeled:
            outliers = np.concatenate([outliers, np.ones((n_outliers, 1))], axis=1)
        inliers = np.concatenate([inliers, outliers], axis=0)

    if normalized:
        inliers = normalize(inliers)

    if randomized:
        rng.shuffle(inliers)

    if enumerated:
        row_num = [i for i in range(1, len(inliers) + 1)]
        inliers = np.insert(inliers, 0, row_num, axis=1)

    np.savetxt(outfile, inliers, delimiter=',', fmt='%1.8f')

    return inliers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--inliers',    '-i', default=950,  type=int, help='number of inliers')
    parser.add_argument('--outliers',   '-o', default=50, type=int, help='number of outliers')
    parser.add_argument('--feats',      '-f', default=2,   type=int, help='number of features')
    parser.add_argument('--blobs',      '-b', default=1,   type=int, help='number of blobs')
    parser.add_argument('--seed',       '-s', default=11,  type=int, help='seed for rng (int)')

    parser.add_argument('--unlabeled',  '-u', action='store_true', help='unlabeled dataset')
    parser.add_argument('--dirty',      '-d', action='store_true', help='some inliers may be labeled as anomalies')
    parser.add_argument('--randomized', '-r', action='store_true', help='shuffle the rows of the dataset')
    parser.add_argument('--enumerated', '-e', action='store_true', help='add feature (column) with row index')
    parser.add_argument('--normalized', '-n', action='store_true', help='normalize to range [0, 1] all feats')

    parser.add_argument('outfile', nargs='?', default=stdout, type=argparse.FileType('w'), help='csv outfile')

    args = parser.parse_args()

    make_dataset(
        n_inliers=args.inliers,
        n_outliers=args.outliers,
        feats=args.feats,
        blobs=args.blobs,
        seed=args.seed,
        unlabeled=args.unlabeled,
        dirty=args.dirty,
        randomized=args.randomized,
        enumerated=args.enumerated,
        normalized=args.normalized,
        outfile=args.outfile
    )
