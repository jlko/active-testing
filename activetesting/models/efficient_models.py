import numpy as np
from types import MethodType


def make_efficient(model, dataset):
    """Model is constant over acquisition.

    Exploit this for efficiency gains.
    Predict on all unobserved test data once at the beginning of training.
    Then, when predict is called, just regurgitate these predictions.

    Currently this does not save stds from predictions.

    If make_efficient is called twice, the model will predict again!
    """
    idxs = dataset.test_remaining
    x = dataset.x[idxs]

    if getattr(model, 'efficient_instance', False):
        if model.cfg.task_type == 'regression':
            out = model.real_predict(x, return_std=True)
        else:
            out = model.real_predict(x)
    else:
        if model.cfg.task_type == 'regression':
            out = model.predict(x, return_std=True)
        else:
            out = model.predict(x)
        model = EfficientModel(model)

    if isinstance(out, tuple):
        # Handle with std
        out = list(out)
        if out[0].ndim == 1:
            preds = np.zeros(dataset.N)
            stds = np.zeros(dataset.N)
        else:
            preds = np.zeros((dataset.N, out[0].shape[1]))
            stds = np.zeros((dataset.N, out[1].shape[1]))

        preds[idxs] = out[0]
        stds[idxs] = out[1]
        model.test_predictions = preds
        model.test_predictions_std = stds
    else:
        if out.ndim == 1:
            preds = np.zeros(dataset.N)
        else:
            preds = np.zeros((dataset.N, out.shape[1]))
        preds[idxs] = out
        model.test_predictions = preds
        model.test_predictions_std = None

    if getattr(model.model, 'has_mi', False):
        mis = np.zeros(dataset.N)
        mi = model.model.predict(x, mutual_info=True)
        mis[idxs] = mi
        model.test_predictions_mi = mis
    else:
        model.test_predictions_mi = None

    return model


class EfficientModel():

    def __init__(self, model):
        self.model = model
        self.cfg = self.model.cfg
        self.efficient_instance = True

    def fit(self, *args, **kwargs):
        if self.cfg.get('keep_constant', False):
            print('debug: no refitting, is efficient')
            pass
        else:
            return self.model.fit(self, *args, **kwargs)

    def real_fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.efficient_predict(*args, **kwargs)

    def real_predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def efficient_predict(
            self, data, idxs, return_std=False, mutual_info=False,
            no_lazy=False, *args, **kwargs):

        if no_lazy:
            self.real_predict(
                data, *args, return_std=return_std,
                mutual_info=mutual_info, **kwargs)

        if return_std and self.test_predictions_std is not None:
            return (self.test_predictions[idxs],
                    self.test_predictions_std[idxs])

        elif mutual_info and self.test_predictions_mi is not None:
            return self.test_predictions_mi[idxs]

        else:
            return self.test_predictions[idxs]
