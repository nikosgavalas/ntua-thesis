"""
This implementation is buggy and slow due to recursion, do not use
"""

import numpy as np


class Branch:

    def __init__(self, position, split_val, split_feat, bbox, left=None,
                 right=None, point=None, parent=None):
        """
        bbox -> bounding box, a numpy 2 x d array where the first row contains
        the mins and the second row the maxes for the nodes below
        """
        self.position = position
        self.split_val = split_val
        self.split_feat = split_feat
        self.bbox = bbox
        self.size = 0
        self.left = left
        self.right = right
        self.point = point
        self.parent = parent


class Leaf:

    def __init__(self, position, point=None, parent=None):
        """
        position: str, 'left' or 'right'
        """
        # HACK
        self.bbox = np.array([point[0], point[0]])
        self.position = position
        self.point = point
        self.parent = parent
        self.size = 1


class RRCT:
    """
    Robust Random Cut Tree
    """

    def __init__(self, dataset, random_state):
        self.ndim = dataset.shape[1]
        self.rng = random_state
        # dictionary used for fast indexing of the leaves, and easy
        # removal and insertion when rrct is used online
        self.leaves_dict = {}
        # filter the dataset, so that it contains no duplicates
        dataset = np.unique(dataset, axis=0)
        self._tree = self.create_tree(dataset)

    def create_tree(self, X, parent=None, position='root'):
        if len(X) > 1:
            maxes = np.max(X, axis=0)
            mins = np.min(X, axis=0)
            ranges = maxes - mins
            sum_ranges = np.sum(ranges)
            if sum_ranges == 0:
                split_feat = self.rng.choice(self.ndim)
            else:
                probability = ranges / sum_ranges
                split_feat = self.rng.choice(self.ndim, p=probability)

            column = X[:, split_feat]
            split_val = self.rng.uniform(np.min(column), np.max(column))
            #print("split at feat {} at value {}".format(split_feat, split_val))

            root = Branch(
                position=position,
                parent=parent,
                split_feat=split_feat,
                split_val=split_val,
                bbox=np.array([mins, maxes])
            )

            s1 = X[column <= split_val]
            root.left = self.create_tree(s1, parent=root, position='left')
            s2 = X[column > split_val]
            root.right = self.create_tree(s2, parent=root, position='right')
            root.size = root.left.size + root.right.size
            self._update_sizes(root, 'inc')
            return root

        new_leaf = Leaf(point=X, parent=parent, position=position)
        self.leaves_dict[len(self.leaves_dict)] = new_leaf
        return new_leaf

    def _update_sizes(self, node, op):
        value = 1 if op == 'inc' else -1
        while node is not None:
            node.size += value
            node = node.parent

    def __str__(self):
        ret = []

        def recur(tree):
            ret.append(tree.point)
            if isinstance(tree, Branch):
                recur(tree.left)
                recur(tree.right)
        recur(self._tree)
        return str(ret)  # or join the list with delimiter of choice?

    def find_leaf_by_value(self, point):
        def recur(tree):
            if isinstance(tree, Leaf):
                if np.equal(tree.point, point).all():
                    return tree
                return None
            if point[tree.split_feat] <= tree.split_val:
                return recur(tree.left)
            return recur(tree.right)
        leaf = recur(self._tree)
        # if leaf is None:
        #    raise ValueError('point does not exist in the tree')
        return leaf

    def find_leaf_by_index(self, leaf_index):
        return self.leaves_dict[leaf_index]

    def score(self, point):
        node = self.find_leaf_by_value(point)
        if node is not None:
            G = []
            while node.parent is not None:
                node = node.parent
                if point[node.split_feat] <= node.split_val:
                    assert node.left.size != 0  # should never be zero
                    G.append(node.right.size / node.left.size)
                else:
                    assert node.right.size != 0
                    G.append(node.left.size / node.right.size)
            return max(G)
        return 0

    def insert_point(self, point):
        def recur(tree):
            assert len(point) == self.ndim
            # TODO: handle case of tree being empty
            mins = np.minimum(tree.bbox[0], point)
            maxes = np.maximum(tree.bbox[1], point)
            r = self.rng.uniform(0, np.sum(maxes - mins))

            partial_sums = 0.0
            split_feat = 0
            for split_feat in range(self.ndim):
                partial_sums += maxes[split_feat] - mins[split_feat]
                if partial_sums >= r:
                    break

            split_val = mins[split_feat] + partial_sums - r
            # if not in range or leaf?
            if (split_val < mins[split_feat] or split_val > maxes[split_feat]
                    or isinstance(tree, Leaf)):
                # TODO bbox and positions?
                new_parent = Branch(
                    None,
                    bbox=np.array([mins, maxes]),
                    split_feat=split_feat,
                    split_val=split_val,
                    parent=tree.parent
                )
                # if tree.parent is None, replace tree with new_parent (??)
                # if tree.parent is None:
                #    return new_parent
                new_parent.position = tree.position
                if point[split_feat] <= split_val:
                    new_leaf = Leaf('left', point=point, parent=new_parent)
                    new_parent.left = new_leaf
                    new_parent.right = tree
                else:
                    new_leaf = Leaf('right', point=point, parent=new_parent)
                    new_parent.right = new_leaf
                    new_parent.left = tree
                self.leaves_dict[len(self.leaves_dict)] = new_leaf
                new_parent.size = tree.size
                self._update_sizes(new_parent, 'inc')
                return new_parent
            else:
                if point[tree.split_feat] <= tree.split_val:
                    tree.left = recur(tree.left)
                else:
                    tree.right = recur(tree.right)
                return tree
        self._tree = recur(self._tree)

    def remove_point_by_value(self, point):
        leaf = self.find_leaf_by_value(point)
        self._remove_leaf(leaf)

    def remove_point_by_index(self, leaf_index):
        leaf = self.find_leaf_by_index(leaf_index)
        self._remove_leaf(leaf)

    def _remove_leaf(self, leaf):
        self._update_sizes(leaf, 'dec')
        father = leaf.parent
        if father is None:
            print('reekris')
            self._tree = None
            self.leaves_dict.clear()
            return
        # father not None:
        sibling = father.left if leaf.position == 'right' else father.right
        grandfather = father.parent
        if grandfather is None:
            sibling.father = None
            self._tree = sibling
            # TODO delete leaf from leaves dict as well (otherwise they can't be garbage-collected!)
            return
        # grandfather not None
        # TODO delete leaf from leaves dict as well
        if father.position == 'left':
            grandfather.left = sibling
        else:
            grandfather.right = sibling
        #gc.collect()


class RRCF:
    """
    Robust Random Cut Forest
    """

    def __init__(self, X, subsample_size=256, num_trees=50, random_state=None):
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random

        self.num_trees = num_trees
        self.ndim = X.shape[1]

        self.ensemble = []
        for _ in range(self.num_trees):
            sample = self._pick_sample(X, subsample_size)
            self.ensemble.append(RRCT(sample, self.rng))

    def _pick_sample(self, dataset, subsample_size):
        if subsample_size > len(dataset):
            return dataset
        return dataset[self.rng.choice(len(dataset), size=subsample_size,
                                       replace=False)]

    def insert_point(self, point):
        for tree in self.ensemble:
            tree.insert_point(point)

    def remove_point(self, index):
        for tree in self.ensemble:
            tree.remove_point_by_index(index)

    def score(self, point):
        score = 0
        for tree in self.ensemble:
            score += tree.score(point)
        return score / self.num_trees
