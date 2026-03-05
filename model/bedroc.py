import numpy as np
from math import exp

def bedroc_score(y_true, y_score, alpha=20.0, empty=-1):
    """
    Compute the BEDROC score for early enrichment evaluation.
    Parameters:
        y_true: array-like, shape (n_samples,)
            True binary labels (0 or 1).
        y_score: array-like, shape (n_samples,)
            Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
        alpha: float
            The steepness parameter for BEDROC (higher = more early enrichment focus).
        empty: int or float
            Value to ignore in y_true (e.g., -1 for missing labels).
    Returns:
        bedroc: float
            The BEDROC score (0-1, higher is better).
    """
    y_true = np.array(y_true).flatten()
    y_score = np.array(y_score).flatten()
    flag = y_true != empty
    y_true = y_true[flag]
    y_score = y_score[flag]
    N = len(y_true)
    n = np.sum(y_true)
    if n == 0 or n == N:
        return np.nan  # undefined if only one class present
    # Rank by descending score
    order = np.argsort(-y_score)
    y_true = y_true[order]
    # Find ranks of positives (1-based)
    ranks = np.where(y_true == 1)[0] + 1
    # BEDROC calculation
    sum_exp = np.sum(np.exp(-alpha * (ranks - 1) / N))
    ra = n / N
    factor1 = (1 - np.exp(-alpha)) / (ra * (1 - np.exp(-alpha * (1 - ra))))
    factor2 = sum_exp / n
    bedroc = factor1 * factor2 + (1 - factor1) / 2
    return bedroc
