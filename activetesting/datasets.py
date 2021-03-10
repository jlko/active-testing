"""Datasets for active testing."""

import logging
from pathlib import Path
import hydra
from omegaconf import OmegaConf

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split as SKtrain_test_split


class _Dataset:
    """Implement generic dataset.

    Load and preprocess data.
    Provide basic acess, train-test split.

    raise generic methods
    """
    def __init__(self, cfg):

        # Set task_type and global_std if not present.
        self.cfg = OmegaConf.merge(
                OmegaConf.structured(cfg),
                dict(
                    task_type=cfg.get('task_type', 'regression'),
                    global_std=cfg.get('global_std', False),
                    n_classes=cfg.get('n_classes', -1)))

        self.N = cfg.n_points
        self.x, self.y = self.generate_data()

        # For 1D data, ensure Nx1 shape
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]

        self.D = self.x.shape[1:]

        self.train_idxs, self.test_idxs = self.train_test_split(self.N)

        if self.cfg.standardize:
            self.standardize()

    def train_test_split(self, N, test_size=None):
        all_indices = np.arange(0, N)

        if self.cfg.get('stratify', False):
            stratify = self.y
        else:
            stratify = None

        if test_size is None:
            test_size = self.cfg.test_proportion

        train, test = SKtrain_test_split(
                all_indices, test_size=test_size,
                stratify=stratify)

        assert np.intersect1d(train, test).size == 0
        assert np.setdiff1d(
            np.union1d(train, test),
            all_indices).size == 0

        if p := self.cfg.get('test_unseen_proportion', False):
            test, test_unseen = SKtrain_test_split(
                np.arange(0, len(test)), test_size=p)
            self.test_unseen_idxs = test_unseen

        return train, test

    @property
    def train_data(self):
        return self.x[self.train_idxs], self.y[self.train_idxs]

    def standardize(self):
        """Standardize to zero mean and unit variance using train_idxs."""

        ax = None if self.cfg['global_std'] else 0

        x_train, y_train = self.train_data

        x_std = self.cfg.get('x_std', True)
        if x_std:
            self.x_train_mean = x_train.mean(ax)
            self.x_train_std = x_train.std(ax)
            self.x = (self.x - self.x_train_mean) / self.x_train_std

        y_std = self.cfg.get('y_std', True)
        if (self.cfg['task_type'] == 'regression') and y_std:
            self.y_train_mean = y_train.mean(ax)
            self.y_train_std = y_train.std(ax)
            self.y = (self.y - self.y_train_mean) / self.y_train_std

    def export(self):
        package = dict(
            x=self.x,
            y=self.y,
            train_idxs=self.train_idxs,
            test_idxs=self.test_idxs
            )
        return package


class _ActiveTestingDataset(_Dataset):
    """Active Testing Dataset.

    Add functionality for active testing.

    Split test data into observed unobserved.

    Add Methods to keep track of unobserved/observed.
    Use an ordered set or sth to keep track of that.
    Also keep track of activation function values at time that
    sth was observed.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.start()

    def start(self):
        self.test_observed = np.array([], dtype=np.int)
        self.test_remaining = self.test_idxs

    def restart(self):
        self.start()

    def observe(self, idx):
        """Observe data at idx and move from unobserved to observed.

        Note: For efficiency reasons idx is index in test
        """
        self.test_observed = np.append(self.test_observed, idx)
        self.test_remaining = self.test_remaining[self.test_remaining != idx]

        return self.x[[idx]], self.y[[idx]]

    @property
    def total_observed(self):
        """Return train and observed test data"""
        test = self.x[self.test_observed], self.y[self.test_observed]
        train = self.train_data
        # concatenate x and y separately
        total_observed = [
            np.concatenate([test[i], train[i]], 0)
            for i in range(2)]

        return total_observed


class QuadraticDatasetForLinReg(_ActiveTestingDataset):
    """Parabolic data for use with linear regression – proof of concept."""
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)

    def generate_data(self):
        x = np.linspace(0, 1, self.N)
        y = x**2
        y -= np.mean(y)
        return x, y


class SinusoidalDatasetForLinReg(_ActiveTestingDataset):
    """Sinusoidal data for use with linear regression – proof of concept.

    This dataset has a high and a low-density region.
    A clever acquisition strategy is necessary to estimate the error correctly.

    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)

    def generate_data(self):
        def regression_function(min_x, max_x, n_samples):
            x = np.linspace(min_x, max_x, n_samples)
            y = np.sin(x * 10) + x ** 3
            return (x, y)

        def split_dataset(n_total, min_x=0, max_x=2, center=1):
            """Split regression function into high and low density regions."""
            n_low = int(0.1 * n_total)
            n_high = int(0.9 * n_total)

            low_density_data = regression_function(
                min_x, center - 0.01, n_low)
            high_density_data = regression_function(
                center, max_x, n_high)

            x = np.concatenate([
                low_density_data[0], high_density_data[0]], 0)
            y = np.concatenate([
                low_density_data[1], high_density_data[1]], 0)

            n_low = len(low_density_data[0])

            return x, y, n_low

        x, y, self.n_low = split_dataset(self.N)

        # TODO: add back!?
        # y = y - np.mean(y)

        return x, y

    def train_test_split(self, *args):
        """Need to overwrite train_test_split.
        Stratify across low and high_density regions.
        """
        n_low = self.n_low
        n_high = self.N - n_low

        low_train, low_test = super().train_test_split(n_low, test_size=4)
        high_train, high_test = super().train_test_split(n_high)
        high_train += n_low
        high_test += n_low

        train = np.concatenate([low_train, high_train], 0)
        test = np.concatenate([low_test, high_test], 0)

        return train, test


class GPDatasetForGPReg(_ActiveTestingDataset):
    """Sample from GP prior."""
    def __init__(self, cfg, model_cfg, *args, **kwargs):
        self.model_cfg = model_cfg
        super().__init__(cfg)

    def generate_data(self):
        from activetesting.utils import maps
        self.model = maps.model[self.model_cfg.name](self.model_cfg)
        xmax = self.cfg.get('xmax', 1)
        x = np.linspace(0, xmax, self.N)[:, np.newaxis]
        y = self.model.sample_y(x, random_state=np.random.randint(0, 10000))
        return x, y


class MNISTDataset(_ActiveTestingDataset):
    """MNIST Data.

    TODO: Respect train/test split of MNIST.
    """
    def __init__(self, cfg, n_classes=10, *args, **kwargs):

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification', global_std=True,
                 n_classes=n_classes))

        super().__init__(cfg)

    def generate_data(self):

        data_home = Path(hydra.utils.get_original_cwd()) / 'data/MNIST'

        # from sklearn.datasets import fetch_openml
        # x, y = fetch_openml(
        #     'mnist_784', version=1, return_X_y=True, data_home=data_home,
        #     cache=True)
        data = keras.datasets.mnist.load_data(
            path=data_home / 'mnist.npz'
        )
        return self.preprocess(data)

    def preprocess(self, data):

        (x_train, y_train), (x_test, y_test) = data
        x = np.concatenate([x_train, x_test], 0)
        x = x.astype(np.float32) / 255
        x = x.reshape(x.shape[0], -1)
        y = np.concatenate([y_train, y_test], 0)
        y = y.astype(np.int)

        N = self.N

        if N < y.size:
            # get a stratified subset
            # note that mnist does not have equal class count
            idxs, _ = SKtrain_test_split(
                np.arange(0, y.size), train_size=N, stratify=y)
            x = x[idxs]
            y = y[idxs]

        return x, y

    def train_test_split(self, N):

        if self.cfg.get('respect_train_test', False):
            train = np.arange(0, 50000)
            n_test = int(self.cfg.test_proportion * 60000)
            test = np.random.choice(
                np.arange(50000, 60000), n_test, replace=False)

            return train, test

        else:
            train, test = super().train_test_split(N)

        # only keep the first n sevens in the train distribution
        if n7 := self.cfg.get('n_initial_7', False):
            # to get correct indices, need to first select from y
            old7 = np.where(self.y == 7)[0]
            # then filter to train indicees
            old_train7 = np.intersect1d(old7, train)
            # now only keep the first n7
            sevens_remove = old_train7[n7:]
            # and now remove those from the train set
            train = np.setdiff1d(train, sevens_remove)

        return train, test


class TwoMoonsDataset(_ActiveTestingDataset):
    """TwoMoons Data."""
    def __init__(self, cfg,
                 *args, **kwargs):

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification', global_std=False, n_classes=2))

        super().__init__(cfg)

    def generate_data(self):

        from sklearn.datasets import make_moons

        x, y = make_moons(n_samples=self.cfg.n_points, noise=self.cfg.noise)

        return x, y


class FashionMNISTDataset(MNISTDataset):
    """FashionMNIST Data.

    TODO: Respect train/test split of FashionMNIST.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg)

    def generate_data(self):

        data = keras.datasets.fashion_mnist.load_data()

        return self.preprocess(data)


class Cifar10Dataset(MNISTDataset):
    """CIFAR10 Data.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg)

    def generate_data(self):

        data = keras.datasets.cifar10.load_data()

        x, y = self.preprocess(data)
        x = x.reshape(len(x), 32, 32, 3).transpose(0, 3, 1, 2)
        x = x.reshape(len(x), -1)
        return x, y[:, 0]


class Cifar100Dataset(MNISTDataset):
    """CIFAR100 Data.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg, n_classes=100)

    def generate_data(self):

        data = keras.datasets.cifar100.load_data()

        x, y = self.preprocess(data)
        x = x.reshape(len(x), 32, 32, 3).transpose(0, 3, 1, 2)
        x = x.reshape(len(x), -1)
        return x, y[:, 0]


def get_CIFAR10():
    """From pruning code. Only used for debugging purposes."""
    import os
    import torch
    from torchvision import transforms, datasets

    root = "./data"

    input_size = 32
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform,
        download=False
    )

    kwargs = {"num_workers": 4, "pin_memory": True}
    batch_size = 128

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader
