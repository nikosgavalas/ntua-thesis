"""
Author: Nick Gavalas <gavalnick@gmail.com>

Implementation of Amazon's Robust Random Cut Forest algorithm for Anomaly
Detection based on Guha et al. paper (http://proceedings.mlr.press/v48/guha16.pdf)

Major thanks to KLabUM whose implementation details of RRCF helped
me alot (https://klabum.github.io/rrcf).

Additionaly to the RRCF, this sourcefile also contains an implementation of
the Isolation Forest scoring algorithm, which, together with the model of the
RRCF can be used to run Isolation Forest in an on-line manner (on-stream).
Details about this scoring method can be found in Liu's et al. paper
(https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)
"""

import numpy as np


class Branch:

    def __init__(self, sfeat, sval, left=None, right=None, parent=None, size=0, bbox=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.sfeat = sfeat
        self.sval = sval
        self.size = size
        self.bbox = bbox


class Leaf:

    def __init__(self, index, depth=None, parent=None, point=None, size=1):
        self.parent = parent
        self.index = index
        self.depth = depth
        self.point = point
        self.size = size
        self.bbox = np.array([point[0], point[0]])  # HACK I guess


class RRCT:

    def __init__(self, ndim, random_state=None):
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random

        self.ndim = ndim
        self.leaves = {}
        self.root = None

    def __str__(self):
        ret = []
        if self.root is None:
            return 'empty'

        def recur(node, prefix, is_tail):
            nonlocal ret
            if isinstance(node, Leaf):
                info = 'index: ({}), point: {}, size: {}'.format(node.index, node.point[0], node.size)
            if isinstance(node, Branch):
                info = '┐feat: {}, value: {}, size: {}'.format(node.sfeat, node.sval, node.size)
            ret.append('{}{}{}\n'.format(prefix, '└──' if is_tail else '├──', info))
            if isinstance(node, Branch):
                recur(node.left, prefix + ('   ' if is_tail else '│  '), False)
                recur(node.right, prefix + ('   ' if is_tail else '│  '), True)

        recur(self.root, '', True)
        return ''.join(ret)

    def _update_sizes_upwards(self, node, inc=1):
        while node:
            node.size += inc
            node = node.parent

    def _update_depths_downwards(self, node, inc=1):
        # TODO not tested
        if isinstance(node, Leaf):
            node.depth += inc
        else:
            self._update_depths_downwards(node.left, inc)
            self._update_depths_downwards(node.right, inc)

    def _calc_bbox(self, bbox1, bbox2):
        bbox = np.vstack([np.minimum(bbox1[0], bbox2[0]),
                          np.maximum(bbox1[1], bbox2[1])])
        return bbox

    def insert(self, point, index):
        assert point.shape[0] == 1
        assert point.shape[1] == self.ndim
        assert index not in self.leaves

        if self.root is None:
            leaf = Leaf(point=point, index=index, depth=0)
            self.root = leaf
            self.ndim = point.shape[1]
            self.leaves[index] = leaf
            return leaf

        node = self.root
        parent = node.parent
        maxdepth = max([leaf.depth for leaf in self.leaves.values()])
        depth = 0
        for _ in range(maxdepth + 1):  # can be replaced with a while True
            bbox = node.bbox

            mins = np.minimum(node.bbox[0], point[0])
            maxes = np.maximum(node.bbox[1], point[0])
            ranges = maxes - mins
            r = self.rng.uniform(0, np.sum(ranges))
            partial_sums = 0.0
            sfeat = 0
            for sfeat in range(self.ndim):
                partial_sums += ranges[sfeat]
                if partial_sums >= r:
                    break
            sval = mins[sfeat] + partial_sums - r

            if sval <= bbox[0][sfeat]:
                leaf = Leaf(point=point, index=index, depth=depth)
                branch = Branch(sfeat=sfeat, sval=sval, left=leaf, right=node, size=(leaf.size + node.size))
                break
            elif sval >= bbox[1][sfeat]:
                leaf = Leaf(point=point, index=index, depth=depth)
                branch = Branch(sfeat=sfeat, sval=sval, left=node, right=leaf, size=(leaf.size + node.size))
                break
            else:
                depth += 1
                if point[0][node.sfeat] <= node.sval:
                    parent = node
                    node = node.left
                    side = 'left'
                else:
                    parent = node
                    node = node.right
                    side = 'right'

        # Set parent of new leaf and old branch
        node.parent = branch
        leaf.parent = branch
        # Set parent of new branch
        branch.parent = parent
        if parent is not None:
            # Set child of parent to new branch
            setattr(parent, side, branch)
        else:
            # If a new root was created, assign the attribute
            self.root = branch

        # Increment depths below branch
        self._update_depths_downwards(branch, inc=1)

        # Increment leaf count above branch
        self._update_sizes_upwards(parent, inc=1)

        # Update (relax) bounding boxes upwards
        bbox = self._calc_bbox(branch.left.bbox, branch.right.bbox)
        branch.bbox = bbox
        node = branch.parent
        while node:
            node.bbox = self._calc_bbox(node.bbox, bbox)
            node = node.parent

        # Add leaf to leaves dict
        self.leaves[index] = leaf

        # Return inserted leaf for convenience
        return leaf

    def remove(self, index):
        assert index in self.leaves

        leaf = self.leaves[index]
        parent = leaf.parent

        # If leaf is root
        if parent is None:
            self.root = None
            self.ndim = None
            return self.leaves.pop(index)

        # If parent is root
        sibling = parent.left if leaf is parent.right else parent.right
        grandparent = parent.parent
        if grandparent is None:
            del parent
            # Set sibling as new root
            sibling.parent = None
            if isinstance(sibling, Leaf):
                sibling.depth = 0
            self.root = sibling
            # Update depths
            self._update_depths_downwards(sibling, inc=-1)
            return self.leaves.pop(index)

        # Else
        if parent is grandparent.left:
            grandparent.left = sibling
        else:
            grandparent.right = sibling
        sibling.parent = grandparent

        # Update depths
        parent = grandparent
        self._update_depths_downwards(sibling, inc=-1)

        # Update leaf counts under each branch
        self._update_sizes_upwards(parent, inc=-1)

        # Update (tighten) the bounding boxes upwards
        node = parent
        while node:
            node.bbox = self._calc_bbox(node.left.bbox, node.right.bbox)
            node = node.parent

        return self.leaves.pop(index)

    def codisp(self, index):
        node = self.leaves[index]
        if node is self.root:
            return 0
        results = []
        while not node.parent is None:
            parent = node.parent
            sibling = parent.left if node is parent.right else parent.right
            results.append(sibling.size / node.size)
            node = parent
        return max(results)

    def query(self, point):
        # TODO not tested
        assert point.shape[0] == 1
        assert point.shape[1] == self.ndim

        def recur(node, point):
            if isinstance(node, Leaf):
                return node
            if point[0][node.sfeat] <= node.sval:
                return recur(node.left, point)
            return recur(node.right, point)

        return recur(self.root, point)

    def pathlen(self, index):
        # TODO not tested
        return self.leaves[index].depth


class Forest:

    def __init__(self, ndim, num_trees, random_state=None):
        self.ndim = ndim
        self._forest = []
        for _ in range(num_trees):
            self._forest.append(RRCT(ndim, random_state=random_state))

    def insert(self, point, index):
        for tree in self._forest:
            tree.insert(point, index)

    def remove(self, index):
        for tree in self._forest:
            tree.remove(index)


class RRCF(Forest):

    def score(self, index):
        score = 0
        for tree in self._forest:
            score += tree.codisp(index)
        return score / len(self._forest)


class IF(Forest):

    def _average_path_length(self, n):
        if n < 2:
            return 1
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    def _anomaly_score(self, mean, window_size):
        return 2 ** (- mean / self._average_path_length(window_size))

    def score(self, index, window_size=None):
        """
        If window_size is not given, the average score (path length) across all
        trees is returned.
        """
        score = 0
        for tree in self._forest:
            score += tree.pathlen(index)

        avg = score / len(self._forest)

        if window_size:
            return self._anomaly_score(avg, window_size) #self.ndim) ??? TODO wtf is happening here? why window_size? (it's the same with the ndim btw) why avg is shit? is it generally shit?
        return avg
