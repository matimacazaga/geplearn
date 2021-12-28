import numpy as np


def jaccard_similarity(X1: np.ndarray, X2: np.ndarray) -> float:

    intersection = (X1 == X2).sum(axis=1)

    return intersection / len(X2)


def cosine_distance(X1: np.ndarray, X2: np.ndarray) -> float:

    numerator = (X1 * X2).sum(axis=1)
    denominator = np.sqrt((X1 ** 2).sum(axis=1) * (X2 ** 2).sum())

    return 1.0 - (numerator / denominator)


def normalised_euclidean_distance(X1: np.ndarray, X2: np.ndarray) -> float:

    X1_norm = X1 / np.sqrt((X1 ** 2).sum(axis=1)).reshape(-1, 1)
    X2_norm = X2 / np.sqrt((X2 ** 2).sum())

    return 1.0 / (1.0 + np.sqrt(((X1_norm - X2_norm) ** 2).sum(axis=1)))
