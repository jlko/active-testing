"""Models for active testing."""

import logging
from omegaconf import OmegaConf
import numpy as np
from scipy.stats import special_ortho_group

from activetesting.loss import RMSELoss, AccuracyLoss, CrossEntropyLoss


class BaseModel:
    """Base class for models."""
    def __init__(self, cfg, model):
        # Set task_type and global_std if not present.
        self.cfg = OmegaConf.merge(
                OmegaConf.structured(cfg),
                dict(task_type=cfg.get('task_type', 'regression'),))

        self.model = model

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

    def performance(self, x, y, task_type):
        pred = self.predict(x)

        if task_type == 'regression':
            logging.info(f'RMSE: {RMSELoss()(pred, y)}')
        elif task_type == 'classification':
            logging.info(f'Accuracy: {AccuracyLoss()(pred, y).mean()}%.')
            logging.info(f'CrossEntropy: {CrossEntropyLoss()(pred, y).mean()}.')
        else:
            raise ValueError


class SKLearnModel(BaseModel):
    """SKLearn derived models."""
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    def fit(self, x, y):
        if x.ndim == 1:
            # Sklearn expects x to be NxD
            x = x[..., np.newaxis]

        self.model = self.model.fit(x, y)

    def predict(self, x, idxs=None, *args, **kwargs):
        # Sklearn expects x to be NxD
        predict_proba = self.cfg['task_type'] == 'classification'

        return self.predict_sk(x, predict_proba=predict_proba, **kwargs)

    def predict_sk(self, x, predict_proba, **kwargs):

        if predict_proba:
            y = self.model.predict_proba(x, **kwargs)
        else:
            y = self.model.predict(x, **kwargs)

        return y


class LinearRegressionModel(SKLearnModel):
    """Simple linear regression."""
    def __init__(self, cfg):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        super().__init__(cfg, model)

    def predict(self, x, idxs=None, *args, **kwargs):
        return self.model.predict(x)


class GaussianProcessRegressor(SKLearnModel):
    """Gaussian Process regression."""
    def __init__(self, cfg):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        k = Matern(length_scale=cfg.get('length_scale', 1.))

        if σ := cfg.get('with_noise', False):
            k += WhiteKernel(noise_level=σ**2)

        model = GaussianProcessRegressor(kernel=k, optimizer=None)

        super().__init__(cfg, model)

    def sample_y(self, x, **kwargs):
        return self.model.sample_y(x, **kwargs)[:, 0]


class RandomForestClassifier(SKLearnModel):
    """Simple linear regression."""
    def __init__(self, cfg):
        from sklearn.ensemble import RandomForestClassifier as SKForest
        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification',))

        sk_args = cfg.get('sk_args', dict())
        model = SKForest(**sk_args)

        super().__init__(cfg, model)


class RandomDirectionRandomForestClassifier(SKLearnModel):
    """Simple linear regression."""
    def __init__(self, cfg, speedup=True, dim=None):
        from sklearn.ensemble import RandomForestClassifier as SKForest
        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification',))

        sk_args = cfg.get('sk_args', dict())
        model = SKForest(**sk_args)
        super().__init__(cfg, model)

        self.speedup = speedup
        if self.speedup:
            # Only sample from ortho group once
            self.n_rots = 40
            self._rotations = special_ortho_group.rvs(dim, size=self.n_rots)

    def fit(self, x, y):
        self.set_rotation()
        rotated = np.dot(x, self.rot)
        return super().fit(rotated, y)

    def set_rotation(self):
        if not self.speedup:
            self.rot = special_ortho_group.rvs(x.shape[1])
        else:
            self.rot = self._rotations[np.random.randint(0, self.n_rots)]

    def predict(self, x, *args, **kwargs):
        rotated = np.dot(x, self.rot)
        return super().predict(rotated, *args, **kwargs)


class SVMClassifier(SKLearnModel):
    """Simple linear regression."""
    def __init__(self, cfg):
        from sklearn.svm import SVC

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification',))

        sk_args = cfg.get('sk_args', dict())
        model = SVC(probability=True, **sk_args)

        super().__init__(cfg, model)


class GPClassifier(SKLearnModel):
    """Simple linear regression."""
    def __init__(self, cfg):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        kernel = Matern(length_scale=1)

        if σ := cfg.get('with_noise', False):
            kernel += WhiteKernel(noise_level=σ**2)

        model = GaussianProcessClassifier(
            kernel=kernel, optimizer=cfg.get('optimizer', None))

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification',))

        super().__init__(cfg, model)
