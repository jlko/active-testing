"""Define losses."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class SELoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Does not aggregate."""
        return (pred-target)**2


class MSELoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Aggregates to single digit."""
        return (SELoss()(pred, target, *args, **kwargs)).mean()


class RMSELoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        """Aggregates to single digit."""
        return np.sqrt(MSELoss()(pred, target, *args, **kwargs))


class AccuracyLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pred, target):
        """Compute 1 - accuracy.

        Expects pred to be probabilities NxC and target to be in [1,..., C].

        Currently inconsistent with Crossentropy loss.

        """
        return 1. - (np.argmax(pred, axis=1) == target).astype(np.float)


class CrossEntropyLoss:

    enc = None
    eps = 1e-15

    def __call__(self, pred, target):
        """Compute Cross-entropy loss.

        TODO: Numerical instabilities?
        pred: Predicted probabilities, NxC
        target: true class values in [1,..., C], N times
        """

        # One-Hot Encode
        if CrossEntropyLoss.enc is None:
            CrossEntropyLoss.enc = OneHotEncoder(sparse=False)
            CrossEntropyLoss.enc.fit(
                np.arange(0, pred.shape[1])[..., np.newaxis])

        # Clipping
        pred = np.clip(pred, self.eps, 1 - self.eps)
        # Renormalize
        pred /= pred.sum(axis=1)[:, np.newaxis]

        one_hot = CrossEntropyLoss.enc.transform(target[..., np.newaxis])
        res = -1 * (one_hot * np.log(pred)).sum(axis=1)

        return res