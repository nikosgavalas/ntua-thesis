## HTTP reduced (3 feats)
### Gauss:
F1 score: 0.9707524000893056
AUC score: 0.9900177657859887
2.566s

### IForest
F1 score: 0.5709852598913887
AUC score: 0.9716686324357183
46.425s

### HSTrees
F1 score: 0.8344370860927153
AUC score: 0.9907538941650855
1m26.458s
{
    'normalized': False,
    'num_trees': 100,
    'size_limit': 500,
    'max_depth': 10,
    'window_size': 5000,
    'est_num_anomalies': 3000
}

### LODA
F1 score: 0.13331724678771792
AUC score: 0.7667553101753763
35.856s
{
    'num_feats': 3,
    'bucket_width': 1,
    'n_vectors': 200,
    'train_size': 30000,
    'prob_sum_border': -8.65
}

## SMTP (Full)
### Gauss (auto)
F1 score: 0.6833333333333333
AUC score: 0.9861654511356532
1.217s

### IForest (auto)
F1 score: 0.9626556016597512
AUC score: 0.964
7.623s

### HSTrees
F1 score: 0.6891891891891891
AUC score: 0.9044055011460721
15.159s
{
    'normalized': False,
    'num_trees': 100,
    'size_limit': 500,
    'max_depth': 10,
    'window_size': 5000,
    'est_num_anomalies': 170
}

### LODA
F1 score: 0.1271232876712329
AUC score: 0.8814828089185247
6.073s
{
    'num_feats': 38,
    'bucket_width': 1,
    'n_vectors': 200,
    'train_size': 1000,
    'prob_sum_border': -4.79
}