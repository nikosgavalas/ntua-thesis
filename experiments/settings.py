
from python.generate_datasets import make_dataset

clusters_datasets = [
    make_dataset(950, 50, 2, blobs=1, labeled=True),
    make_dataset(950, 50, 2, blobs=2, labeled=True),
    make_dataset(950, 50, 2, blobs=3, labeled=True),
    make_dataset(950, 50, 2, blobs=4, labeled=True),
]

clusters_algo_args = [
    {
        'iter': 1000
    },
    {
        'use_cont': False,
        'cont': 0.05,
        'num_trees': 100,
        'subsample_size': 256
    },
    {
        'normalized': False,
        'num_trees': 50,
        'size_limit': 25,
        'max_depth': 5,
        'window_size': 256,
        'est_num_anomalies': 50
    },
    {
        'num_feats': 2,
        'bucket_width': 1,
        'n_vectors': 100,
        'train_size': 500,
        'prob_sum_border': -4.1
    }
]

clusters_values = [1, 2, 3, 4]

# 5% outliers
examplesize_datasets = [
    make_dataset(950, 50, 5, labeled=True),
    make_dataset(19760, 1040, 5, labeled=True),
    make_dataset(38570, 2030, 5, labeled=True),
    make_dataset(57380, 3020, 5, labeled=True),
    make_dataset(76190, 4010, 5, labeled=True),
    make_dataset(95000, 5000, 5, labeled=True),
]

examplesize_algo_args = [
    {
        'iter': 100
    },
    {
        'use_cont': False,
        'cont': 0.05,
        'num_trees': 100,
        'subsample_size': 256
    },
    {
        'normalized': False,
        'num_trees': 50,
        'size_limit': 25,
        'max_depth': 5,
        'window_size': 256,
        'est_num_anomalies': 50
    },
    {
        'num_feats': 5,
        'bucket_width': 1,
        'n_vectors': 200,
        'train_size': 500,
        'prob_sum_border': -4.5
    }
]

# linespace
examplesize_values = [950, 19760, 38570, 57380, 76190, 95000]

# 5% outliers
featuresize_datasets = [
    make_dataset(950, 50, 5, labeled=True),
    make_dataset(950, 50, 24, labeled=True),
    make_dataset(950, 50, 43, labeled=True),
    make_dataset(950, 50, 62, labeled=True),
    make_dataset(950, 50, 81, labeled=True),
    make_dataset(950, 50, 100, labeled=True),
]

# num_feats of LODA must equal the dataset's!
featuresize_algo_args = [
    {
        'iter': 1000
    },
    {
        'use_cont': False,
        'cont': 0.05,
        'num_trees': 100,
        'subsample_size': 256
    },
    {
        'normalized': False,
        'num_trees': 50,
        'size_limit': 25,
        'max_depth': 5,
        'window_size': 256,
        'est_num_anomalies': 50
    },
    {
        'num_feats': 5,
        'bucket_width': 2,
        'n_vectors': 200,
        'train_size': 500,
        'prob_sum_border': -4.5
    }
]

# linespace
featuresize_values = [5, 24, 43, 62, 81, 100]
