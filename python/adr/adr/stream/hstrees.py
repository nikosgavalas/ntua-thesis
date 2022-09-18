"""
Author: Nick Gavalas <gavalnick@gmail.com>

Implementation of Half-Space Trees algorithm for Anomaly Detection based on
Tan's paper https://www.ijcai.org/Proceedings/11/Papers/254.pdf
"""

# TODO: update the model after λ detected changes, not immediately.
# NOTE TODO: NOT TESTED

import numpy as np


class Leaf:

    def __init__(self, parent=None, size=0, depth=0):
        self.parent = parent
        self.size = size
        self.depth = depth


class Branch:

    def __init__(self, sfeat, sval, left=None, right=None, parent=None,
                 size=0, depth=0):
        self.sfeat = sfeat
        self.sval = sval
        self.left = left
        self.right = right
        self.parent = parent
        self.size = size
        self.depth = depth


class HST:

    # d_min,d_max: arrays of given minimum and maximum values for each dimension
    # recommended hyperparameter values
    def __init__(self, d_min, d_max, num_trees=25, size_limit=25, max_depth=15,
                 random_state=None):
        assert d_max.shape == d_min.shape
        assert d_max.shape[0] == 1

        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random

        self.ndim = len(d_max[0])
        self.num_trees = num_trees
        self.size_limit = size_limit
        self.max_depth = max_depth

        # Dictionary that maps an inserted point to its corresponding Leaf
        self.leaves = {}

        r_min, r_max = self._init_working_space(d_min, d_max)
        self.root = self._build_tree(r_min, r_max)

    def _init_working_space(self, d_min, d_max):
        r_min = np.zeros(self.ndim)
        r_max = np.zeros(self.ndim)

        for q in range(self.ndim):
            s = self.rng.uniform(d_min[0][q], d_max[0][q])
            sigma = 2 * max(s - d_min[0][q], d_max[0][q] - s)
            r_min[q] = s - sigma
            r_max[q] = s + sigma

        return r_min, r_max

    def _build_tree(self, r_min, r_max):
        def recur(r_min, r_max, curr_depth, curr_node):
            if curr_depth == self.max_depth:
                return Leaf(parent=curr_node, depth=curr_depth)

            sfeat = self.rng.choice(self.ndim)
            sval = (r_min[sfeat] + r_max[sfeat]) / 2

            branch = Branch(sfeat, sval, parent=curr_node, depth=curr_depth)

            temp = r_max[sfeat]
            r_max[sfeat] = sval
            branch.left = recur(r_min, r_max, curr_depth + 1, branch)
            r_max[sfeat] = temp

            r_min[sfeat] = sval
            branch.right = recur(r_min, r_max, curr_depth + 1, branch)

            return branch
        return recur(r_min, r_max, 0, None)

    def insert(self, point, index):
        assert point.shape[0] == 1
        assert point.shape[1] == self.ndim
        assert index not in self.leaves

        def recur(node):
            node.size += 1
            self.leaves[index] = node

            if node.depth < self.max_depth:
                if point[0][node.sfeat] < node.sval:
                    recur(node.left)
                else:
                    recur(node.right)
        recur(self.root)

    def remove(self, index):
        assert index in self.leaves
        node = self.leaves[index]
        while node:
            node.size -= 1
            node = node.parent
        del self.leaves[index]

    def query(self, point):
        """ Find the leaf corresponding to a given point """
        def recur(node):
            if isinstance(node, Leaf):
                return node
            if point[node.sfeat] < node.sval:
                return recur(node.left)
            return recur(node.right)
        return recur(self.root)

    def _score(self, node):
        return node.size * (2 ** node.depth)

    def score(self, arg):
        if isinstance(arg, int):
            assert arg in self.leaves
            leaf = self.leaves[arg]
            return self._score(leaf)
        if isinstance(arg, np.ndarray):
            def recur(node):
                if node.depth == self.max_depth or node.size < self.size_limit:
                    return self._score(node)
                if arg[0][node.sfeat] < node.sval:
                    return recur(node.left)
                return recur(node.right)
            return recur(self.root)
        else:
            raise ValueError('instance of int or numpy.ndarray expected')


class HSF:

    def __init__(self, d_min, d_max, num_trees=25, size_limit=25,
                 max_depth=15, random_state=None):
        self._forest = []
        for _ in range(num_trees):
            self._forest.append(HST(d_min, d_max, size_limit=size_limit,
                                    max_depth=max_depth,
                                    random_state=random_state))

    def insert(self, point, index):
        for tree in self._forest:
            tree.insert(point, index)

    def remove(self, index):
        for tree in self._forest:
            tree.remove(index)

    def score(self, index):
        scores = [tree.score(index) for tree in self._forest]
        return sum(scores) / len(scores)  # Take the average value for the sum, otherwise the score values are huge.
