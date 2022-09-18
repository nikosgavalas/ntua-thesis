from collections import deque
import numpy as np


def rolling_window(sequence, size):
    """
    Usage example:
    for el in rolling_window(list(range(20)), 5):
        print(el)
    """
    iterator = iter(sequence)
    init = (next(iterator) for _ in range(size))
    window = deque(init, maxlen=size)
    if len(window) < size:
        raise IndexError('Sequence smaller than window size')
    yield np.asarray(window)
    for elem in iterator:
        window.append(elem)
        yield np.asarray(window)


def normalize(dataset):
    """
    Minmax normalization
    """
    dataset -= np.min(dataset, axis=0)
    dataset = dataset / np.ptp(dataset, axis=0)
    return dataset
