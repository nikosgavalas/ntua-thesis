"""
Author: Nick Gavalas <gavalnick@gmail.com>

Implementation of Isolation Forest based on Liu's at al. paper
(https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
"""

import json
import numpy as np


class Leaf:

    def __init__(self, size):
        self.size = size


class Branch:

    def __init__(self, left_node, right_node, split_feat, split_value):
        self.left_node = left_node
        self.right_node = right_node
        self.split_feat = split_feat
        self.split_value = split_value


class IsolationForest:

    def __init__(self, num_trees=100, subsample_size=256, seed=0xdeadbabe):
        self.num_trees = num_trees
        self.subsample_size = subsample_size

        self.forest = []
        self.height_limit = np.ceil(np.log2(subsample_size))

        self.EULERS_CONST = 0.5772156649

        self.rng = np.random.RandomState(seed)

    def _pick_sample(self, dataset):
        if self.subsample_size > len(dataset):
            return dataset

        return dataset[self.rng.choice(len(dataset), size=self.subsample_size, replace=False)]

    def fit(self, dataset):
        for _ in range(self.num_trees):
            sample = self._pick_sample(dataset)
            self.forest.append(self._build_iTree(sample, 0))

    def _build_iTree(self, subset, curr_height):
        if curr_height >= self.height_limit or len(subset) <= 1:
            return Leaf(len(subset))

        # randomly select a feature column
        split_feat = self.rng.choice(subset.shape[1])
        column = subset[:, split_feat]

        # split at a random point between two marginal values
        split_value = self.rng.uniform(column.min(), column.max())
        left_subset = subset[column < split_value]
        right_subset = subset[column >= split_value]

        return Branch(
            left_node=self._build_iTree(left_subset, curr_height + 1),
            right_node=self._build_iTree(right_subset, curr_height + 1),
            split_feat=split_feat,
            split_value=split_value
        )

    def _average_path_length(self, n):
        if n < 2:
            return 1

        return 2 * (np.log(n - 1) + self.EULERS_CONST) - (2 * (n - 1) / n)

    def _anomaly_score(self, mean):
        return 2 ** (- mean / self._average_path_length(self.subsample_size))

    def _path_length(self, instance, tree, curr_path_length):
        if isinstance(tree, Leaf):
            return curr_path_length + self._average_path_length(tree.size)

        if instance[tree.split_feat] < tree.split_value:
            return self._path_length(instance, tree.left_node, curr_path_length + 1)

        return self._path_length(instance, tree.right_node, curr_path_length + 1)

    def score(self, dataset):
        """
        Flag as anomaly if the score is > 0.5
        """
        # pre-allocating the array yields faster performance
        ret = np.empty(len(dataset))

        # this 'for' is parallelizable
        for idx, example in enumerate(dataset):

            heights_sum = 0

            for tree in self.forest:
                heights_sum += self._path_length(example, tree, 0)

            # calculate the mean
            E_val = heights_sum / self.num_trees

            ret[idx] = self._anomaly_score(E_val)

        return ret

    def predict(self, dataset, contamination=-1):
        ret = np.zeros(len(dataset))

        scores = self.score(dataset)

        if 0 < contamination < 1:
            # contamination is the percentage of outliers
            # flag the top m as anomalies
            m = int(np.ceil(len(dataset) * contamination))
            ret[scores.argsort()[-m:]] = 1
        else:
            ret[scores > 0.5] = 1

        return ret

    def __str__(self):
        return self._model_to_string()

    def _serialize_tree(self, tree, lst):
        if isinstance(tree, Leaf):
            lst.append({
                't': 'e',
                's': tree.size
            })
            return

        lst.append({
            'f': tree.split_feat,
            'v': tree.split_value,
            't': 'i'
        })

        self._serialize_tree(tree.left_node, lst)
        self._serialize_tree(tree.right_node, lst)

    def _deserialize_tree(self, tree_array):
        curr = tree_array.pop(0)

        if curr['t'] == 'e':
            return Leaf(curr['s'])

        return Branch(self._deserialize_tree(tree_array), self._deserialize_tree(tree_array), curr['f'], curr['v'])

    def _model_to_string(self):
        ret = {
            'model': [],
            'size': self.subsample_size
        }

        for tree in self.forest:
            lst = []
            self._serialize_tree(tree, lst)
            ret['model'].append(lst)

        return json.dumps(ret, separators=(',', ':'))

    def _model_from_string(self, string):
        self.forest = []

        obj = json.loads(string)
        self.subsample_size = obj['size']
        model = obj['model']
        for tree_array in model:
            self.forest.append(self._deserialize_tree(tree_array))

    def save_to_disk(self, filename='model.json'):
        with open(filename, 'w') as f:
            f.write(self._model_to_string())

    def load_from_disk(self, filename='model.json'):
        with open(filename, 'r') as f:
            serialized_model = f.read()
        self._model_from_string(serialized_model)
