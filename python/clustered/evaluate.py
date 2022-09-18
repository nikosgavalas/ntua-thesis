#!/usr/bin/env python3

from sys import stdin

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score


def evaluate(unlabeled=False, enumerated=False, scored=False, outliers=None, threshold=None, infile=stdin,
             plotfile=None):
    # first load the whole dataset
    data = np.genfromtxt(infile, delimiter=',')

    if enumerated:
        # drop the first column with the indices
        data = np.delete(data, 0, 1)

    scores = None
    if scored:
        scores = data[:, -1]
        data = data[:, :-1]

    labels = None
    if not unlabeled:
        labels = data[:, -1]
        data = data[:, :-1]

    feats = data.shape[1]

    if feats == 2:
        plt.scatter(data[:, 0], data[:, 1], color='blue', s=10)
        if not unlabeled:
            outliers_data = data[labels == 1.0]
            plt.scatter(outliers_data[:, 0], outliers_data[:, 1], color='orange', s=10)

    if not unlabeled and scored:
        print(f'rocauc:{roc_auc_score(1 - labels, scores)}')
        if threshold or outliers:
            if outliers and not threshold:
                srt = np.sort(scores)
                threshold = srt[int(outliers)]
            preds = scores < threshold
            print(f'f1:{f1_score(labels, preds)}')
            if feats == 2:
                labels_bool = labels.astype(bool)
                correctly_classified = data[labels_bool ^ preds == 0.0]
                incorrectly_classified = data[labels_bool ^ preds == 1.0]
                plt.scatter(correctly_classified[:, 0], correctly_classified[:, 1], color='green', s=10)
                plt.scatter(incorrectly_classified[:, 0], incorrectly_classified[:, 1], color='red', s=10)

    if feats == 2:
        if plotfile:
            plt.savefig(plotfile)
        else:
            plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # dataset format
    parser.add_argument('--enumerated', '-e', action='store_true', help='first column is the row index')
    parser.add_argument('--unlabeled',  '-u', action='store_true', help='unlabeled dataset')
    parser.add_argument('--scored',     '-s', action='store_true', help='last column is score predicted by a model')

    # threshold - estimation
    parser.add_argument('--outliers',   '-o', help='number of outliers (to estimate threshold)')
    parser.add_argument('--threshold',  '-t', help='hard threshold value')

    parser.add_argument('--plotfile',   '-p', help='save plot in file with given name')

    parser.add_argument('infile', nargs='?', default=stdin, type=argparse.FileType('r'), help='csv infile')

    args = parser.parse_args()

    evaluate(
        unlabeled=args.unlabeled,
        enumerated=args.enumerated,
        scored=args.scored,
        outliers=args.outliers,
        threshold=args.threshold,
        infile=args.infile,
        plotfile=args.plotfile
    )
