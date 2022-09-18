#!/usr/bin/env python3

from sys import stdin, stdout, stderr

import numpy as np

from adr.stream.gauss import MultivariateGaussian
from adr.stream.hstrees import HSF
from adr.stream.loda import LODA
from adr.stream.rrcf import RRCF, IF

# TODO put the four methods in a class and separate it from the arg parser, so that it can be used programmatically without a cli


def _run_model(args, model):
    for index, line in enumerate(args.infile):
        raw = line.rstrip().split(',')
        data_str = raw.copy()
        if args.enumerated:
            data_str = data_str[1:]
        if not args.unlabeled:
            data_str = data_str[:-1]
        data = np.array(data_str, dtype=np.float64).reshape(1, args.feats)
        model.insert(data, index)
        score = model.score(index) if index > args.train else 1
        raw += [str(score)]
        args.outfile.write(','.join(raw) + '\n')


def gauss(args):
    model = MultivariateGaussian(ndim=args.feats)

    _run_model(args, model)


# using underscore because it conflicts with 'if' token
def if_(args):
    n_trees = args.n

    model = IF(args.f, n_trees)

    _run_model(args, model)


def hsf(args):
    n_trees = args.n
    size_limit = args.s
    max_depth = args.d

    dmin = -13 * np.ones(args.f)
    dmax = 13 * np.ones(args.f)
    model = HSF(dmin, dmax, n_trees, size_limit, max_depth)

    _run_model(args, model)


def loda(args):
    nvec = args.n
    bwidth = args.w

    model = LODA(args.f, bwidth, nvec)

    _run_model(args, model)


def rrcf(args):

    model = RRCF(args.f, args.n)

    _run_model(args, model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile',     '-i', default=stdin,  type=argparse.FileType('r'), help='csv infile')
    parser.add_argument('--outfile',    '-o', default=stdout, type=argparse.FileType('w'), help='csv outfile')
    parser.add_argument('--unlabeled',  '-u', action='store_true', help='unlabeled data')
    parser.add_argument('--enumerated', '-e', action='store_true', help='enumerated lines')
    parser.add_argument('--feats',      '-f', default=2,     type=int, help='number of features')
    parser.add_argument('--train',      '-t', default=20,    type=int, help='number of instances used for training')
    # TODO add seed

    subparsers = parser.add_subparsers()

    gauss_parser = subparsers.add_parser('gauss')
    gauss_parser.set_defaults(func=gauss)

    if_parser = subparsers.add_parser('if')
    if_parser.add_argument('--n-trees', '-n', default=50, type=int, help='number of trees')
    if_parser.set_defaults(func=if_)

    hsf_parser = subparsers.add_parser('hsf')
    hsf_parser.add_argument('--n-trees', '-n', default=100,   type=int, help='number of trees')
    hsf_parser.add_argument('--size',    '-s', default=10000, type=int, help='size limit')
    hsf_parser.add_argument('--depth',   '-d', default=18,    type=int, help='max depth')
    hsf_parser.set_defaults(func=hsf)

    loda_parser = subparsers.add_parser('loda')
    loda_parser.add_argument('--n-vectors', '-n', default=200, type=int,   help='number of vectors')
    loda_parser.add_argument('--width',     '-w', default=1.0, type=float, help='bucket width')
    loda_parser.set_defaults(func=loda)

    rrcf_parser = subparsers.add_parser('rrcf')
    rrcf_parser.add_argument('--n-trees', '-n', default=30, type=int, help='number of trees')
    rrcf_parser.set_defaults(func=rrcf)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
        exit(1)
    args.func(args)
