import numpy as np
from sklearn.metrics import f1_score


class MultivariateGaussian:

    def __init__(self):
        self.mean = np.empty(0)
        self.variance = np.empty(0)

    def fit(self, dataset):
        self.mean = dataset.mean(axis=0)  # , dtype=np.float64)
        self.variance = dataset.var(axis=0)

    def score(self, dataset):
        nom = np.exp(- np.power(dataset - self.mean, 2) / (2 * self.variance))
        denom = np.sqrt(2 * np.pi * self.variance)
        return np.prod(nom / denom, axis=1)

    def predict(self, probabilities, threshold):
        ret = np.zeros(len(probabilities))
        ret[probabilities < threshold] = 1
        return ret


def suggest_threshold(gauss_model, eval_dataset, labels, iterations=1000):
    """ Find optimal threshold setting (that maximizes the f1 score) """
    scores = gauss_model.score(eval_dataset)

    lower = np.min(scores)
    upper = np.max(scores)

    opt = 0
    max_score = 0

    for threshold in np.linspace(lower, upper, num=iterations):
        preds = gauss_model.predict(scores, threshold)
        if (preds == 0).all():
            continue
        score = f1_score(labels, preds)
        if score > max_score:
            max_score = score
            opt = threshold

    return opt
