"""Implement acquisition functions."""

import warnings
import numpy as np
from omegaconf import OmegaConf

from activetesting.models import (
    SVMClassifier, RandomForestClassifier, GPClassifier,
    GaussianProcessRegressor, RandomDirectionRandomForestClassifier,
    RadialBNN, make_efficient
    )


class AcquisitionFunction:
    """Acquisition function is its own class.

    In the beginning this may seem like overkill, but remember that our
    acquisition function will likely have a powerfull substitute model.

    Implement get_next_point
    """
    def __init__(self, cfg_run, dataset):
        self.cfg, run = cfg_run
        self.dataset = dataset
        # keep track of acquisition weights
        self.weights = np.array([])

        if self.cfg.animate and run < self.cfg.animate_until:
            self.all_pmfs = list()
        else:
            self.all_pmfs = None

        self.counter = 0

        if self.cfg.lazy_save:

            if L := self.cfg.get('lazy_save_schedule', False):
                L = list(L)
            else:
                L = list(range(1000))
                L += list(range(int(1e3), int(1e4), 500))
                L += list(range(int(1e4), int(10e4), int(1e3)))

            self.lazy_list = L

        # For model selection hot-patching.
        self.externally_controlled = False
        self.ext_test_idx = None
        self.ext_pmf_idx = None

    @staticmethod
    def acquire():
        raise NotImplementedError

    def check_save(self, off=0):
        if self.all_pmfs is None:
            return False
        if self.cfg.lazy_save and (self.counter - off in self.lazy_list):
            return True
        else:
            return False

        return True

    def sample_pmf(self, pmf):
        """Sample from pmf."""

        if len(pmf) == 1:
            # Always choose last datum
            pmf = [1]

        if self.externally_controlled:
            idx = self.ext_pmf_idx
            test_idx = self.ext_test_idx

        else:
            if self.cfg['sample']:
                # this is one-hot over all remaining test data
                sample = np.random.multinomial(1, pmf)
                # idx in test_remaining
                idx = np.where(sample)[0][0]
            else:
                idx = np.argmax(pmf)

            # get index of chosen test datum
            test_idx = self.dataset.test_remaining[idx]

        # get value of acquisition function at that index
        self.weights = np.append(
            self.weights, pmf[idx])

        if self.check_save():
            self.all_pmfs.append(dict(
                idx=idx,
                test_idx=test_idx,
                pmf=pmf,
                remaining=self.dataset.test_remaining,
                observed=self.dataset.test_observed))

        self.counter += 1
        return test_idx, idx

    @staticmethod
    def safe_normalise(pmf):
        """If loss is 0, we want to sample uniform and avoid nans."""

        if (Σ := pmf.sum()) != 0:
            pmf /= Σ
        else:
            pmf = np.ones(len(pmf))/len(pmf)

        return pmf


class RandomAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset)

    def acquire(self):
        n_remaining = len(self.dataset.test_remaining)
        pmf = np.ones(n_remaining)/n_remaining
        return self.sample_pmf(pmf)


class TrueLossAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, true_loss_vals, *args, **kwargs):
        super().__init__(cfg, dataset)

        # make sure indexes are aligned
        self.true_loss = np.zeros(dataset.N)
        self.true_loss[self.dataset.test_idxs] = true_loss_vals

    def acquire(self):
        """Sample according to true loss dist."""

        pmf = self.true_loss[self.dataset.test_remaining]

        pmf = self.safe_normalise(pmf)

        return self.sample_pmf(pmf)


class DistanceBasedAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset)

    def acquire(self):
        """Sample according to distance to previously sampled points."""
        remaining_idx = self.dataset.test_remaining
        observed_idx = self.dataset.test_observed

        # First test index sampled at random
        if observed_idx.size == 0:
            N = len(self.dataset.test_idxs)
            pmf = np.ones(N) / N

        else:
            # For each point in remaining
            # calculate distance to all points in observed
            remaining = self.dataset.x[remaining_idx]
            observed = self.dataset.x[observed_idx]

            # broadcasting to get all paired differences
            d = remaining[:, np.newaxis, :] - observed
            d = d**2
            # sum over feature dimension
            d = d.sum(-1)
            # sqrt to get distance
            d = np.sqrt(d)
            # mean over other pairs
            distances = d.mean(1)

            # Constract PDF via softmax
            pmf = np.exp(distances)
            pmf /= pmf.sum()

        return self.sample_pmf(pmf)


# --- Acquisition Functions Based on Expected Loss

class _LossAcquisitionBase(AcquisitionFunction):
    def __init__(self, cfg, dataset, model):
        super().__init__(cfg, dataset)

        # also save original model
        self.model = model

    def acquire(self):
        # predict + std for both models on all remaining test points
        remaining_idxs = self.dataset.test_remaining
        remaining_data = self.dataset.x[remaining_idxs]

        # build expected loss
        expected_loss = self.expected_loss(remaining_data, remaining_idxs)

        if self.cfg['sample'] and (expected_loss < 0).sum() > 0:
            # Log-lik can be negative.
            # Make all values positive.
            # Alternatively could set <0 values to 0.
            expected_loss += np.abs(expected_loss.min())

        if not (expected_loss.sum() == 0):
            expected_loss /= expected_loss.sum()

        if self.cfg.get('uniform_clip', False):
            # clip all values less than 10 percent of uniform propability
            p = self.cfg['uniform_clip_val']
            expected_loss = np.maximum(p * 1/expected_loss.size, expected_loss)
            expected_loss /= expected_loss.sum()

        return self.sample_pmf(expected_loss)


class GPAcquisitionUncertainty(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, **kwargs):

        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):

        mu, std = self.model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)

        aleatoric = getattr(self.dataset, 'aleatoric', 0)

        return std**2 + aleatoric**2


class BNNClassifierAcquisitionMI(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, **kwargs):

        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):

        mutual_information = self.model.predict(
            remaining_data, idxs=remaining_idxs, mutual_info=True)

        return mutual_information


class _SurrogateAcquisitionBase(_LossAcquisitionBase):
    def __init__(self, cfg_run, dataset, model, SurrModel, surr_cfg):
        if surr_cfg.get('acquisition', False):
            # the surrogate acquisition can specialise the
            # acquisition configs. this mostly affects clipping behaviour
            cfg = OmegaConf.merge(
                OmegaConf.structured(cfg_run[0]),
                OmegaConf.structured(surr_cfg.acquisition))
            cfg_run = [cfg, cfg_run[1]]

        super().__init__(cfg_run, dataset, model)

        self.surr_cfg = surr_cfg
        self.surr_class = SurrModel

        self.surr_model = SurrModel(surr_cfg)
        self.surr_model.fit(*self.dataset.total_observed)

        if self.surr_cfg.get('efficient', False):
            # make efficient predictions on remaining test data
            self.surr_model = make_efficient(self.surr_model, self.dataset)

        if surr_cfg.get('lazy', False):
            if (sched := surr_cfg.get('lazy_schedule', False)) is not False:
                retrain = list(sched)
            else:
                retrain = [5]
                retrain += list(range(10, 50, 10))
                retrain += [50]
                retrain += list(range(100, 1000, 150))
                retrain += list(range(1000, 10000, 2000))
                retrain += list(range(int(10e3), int(100e3), int(10e3)))

            # always remove 0, since we train at it 0
            self.retrain = list(set(retrain) - {0})
            self.update_surrogate = self.lazy_update_surrogate
        else:
            self.update_surrogate = self.vanilla_update_surrogate

    def vanilla_update_surrogate(self):
        # train surrogate on train data + currently observed test
        self.surr_model = self.surr_class(self.surr_cfg)

        if self.surr_cfg.get('on_train_only', False):
            self.surr_model.fit(*self.dataset.train_data)
        else:
            # fit on all observed data
            self.surr_model.fit(*self.dataset.total_observed)

        if self.surr_cfg.get('efficient', False):
            self.surr_model = make_efficient(self.surr_model, self.dataset)

    def lazy_update_surrogate(self):

        if self.counter in self.retrain:
            self.surr_model = self.surr_class(self.surr_cfg)

            if self.surr_cfg.get('on_train_only', False):
                self.surr_model.fit(*self.dataset.train_data)
            else:
                # fit on all observed data
                self.surr_model.fit(*self.dataset.total_observed)

            if self.surr_cfg.get('efficient', False):
                # make efficient predictions on remaining test data
                self.surr_model = make_efficient(self.surr_model, self.dataset)

    def acquire(self):

        self.update_surrogate()

        return super().acquire()


class _SelfSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        from activetesting.utils.maps import model as model_maps
        SurrModel = model_maps[model.cfg['name']]

        super().__init__(cfg, dataset, model, SurrModel, model_cfg)


class SelfSurrogateAcquisitionEntropy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class SelfSurrogateAcquisitionAccuracy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return accuracy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class _AnySurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        from activetesting.utils.maps import model as model_maps
        SurrModel = model_maps[model_cfg.name]
        super().__init__(cfg, dataset, model, SurrModel, model_cfg)


class AnySurrogateAcquisitionEntropy(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class AnySurrogateAcquisitionAccuracy(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return accuracy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class _GPSurrogateAcquisitionBase(_SurrogateAcquisitionBase):
    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(
            cfg, dataset, model, GaussianProcessRegressor, model_cfg)


class GPSurrogateAcquisitionLogLik(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionLogLik only works if aleatoric noise 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs, *args, **kwargs):

        std = dict(return_std=True)
        mu_s, std_s = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, **std)
        mu_m, std_m = self.model.predict(
            remaining_data, idxs=remaining_idxs, **std)

        # temporary fix for log lik acquisition
        aleatoric = 0

        expected_loss = (
            np.log(2*np.pi*std_m**2)
            + 1/(2*std_m**2) * (
                (mu_s - mu_m)**2 + std_s**2 + aleatoric**2)
        )

        return expected_loss


class GPSurrogateAcquisitionMSE(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionMSE is currently only appropriate if '
        'the aleatoric uncertainty is 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        mu_s, std_s = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)
        mu_m = self.model.predict(remaining_data, idxs=remaining_idxs)

        # each model needs to have this set
        # TODO: should probs be a data property
        # (move reliance on main model out of GPDataset)
        aleatoric = getattr(self.dataset, 'aleatoric', 0)

        expected_loss = (mu_s - mu_m)**2 + std_s**2 + aleatoric**2

        # print('mse/var', (((mu_s - mu_m)/std_s)**2).mean(),
        #       ((mu_s - mu_m)**2).std(), (std_s**2).std())

        if self.cfg.get('clip', False):
            clip_val = 0.05 * np.max(expected_loss)
            if clip_val < 1e-10:
                warnings.warn('All loss values small!')

            expected_loss = np.maximum(clip_val, expected_loss)

        return expected_loss


class GPSurrogateAcquisitionMSEDoublyUncertain(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionMSEDoublyUncertain is currently only '
        'appropriate if the aleatoric uncertainty is 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)
        self.model_cfg = model_cfg

    def expected_loss(self, remaining_data, remaining_idxs):

        mu_s, std_s = self.surr_model.predict(remaining_data, return_std=True)
        mu_m, std_m = self.model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)

        expected_loss = (mu_s - mu_m)**2 + std_s**2 + std_m**2

        return expected_loss


class ClassifierAcquisitionEntropy(_LossAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model)
        self.T = model_cfg.get('temperature', None)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, None, T=self.T,
            cfg=self.cfg)


class ClassifierAcquisitionAccuracy(_LossAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):
        return accuracy_loss(
            remaining_data, remaining_idxs, self.model, None)


class _RandomForestSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(
            cfg, dataset, model, RandomForestClassifier, model_cfg)


class RandomForestClassifierSurrogateAcquisitionEntropy(
        _RandomForestSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class _SVMClassifierSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(cfg, dataset, model, SVMClassifier, model_cfg)


class SVMClassifierSurrogateAcquisitionEntropy(
        _SVMClassifierSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class _GPClassifierSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(cfg, dataset, model, GPClassifier, model_cfg)


class GPClassifierSurrogateAcquisitionEntropy(
        _GPClassifierSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg
            )


class _RandomRandomForestSurrogateAcquisitionBase(_LossAcquisitionBase):
    """Randomize Hypers each iteration."""

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(cfg, dataset, model)

        self.model_cfg = model_cfg
        self.surr_model = None
        self.random_init_model()

    def random_init_model(self):

        if self.model_cfg['params_from'] == 'main':
            if self.surr_model is not None:
                return True
            else:
                sk_args = self.model.model.get_params()
                cfg = OmegaConf.create(dict(sk_args=sk_args))

        elif self.model_cfg['params_from'] == 'random':
            # This may be highly dependent on the data!!
            sk_args = dict(
                max_features='sqrt',
                criterion=str(np.random.choice(["gini", "entropy"])),
                max_depth=int(np.random.choice([3, 5, 10, 20])),
                n_estimators=int(np.random.choice([10, 50, 100, 200])),
                # min_samples_split=int(np.random.choice([2, 5, 10]))
            )
            cfg = OmegaConf.create(dict(sk_args=sk_args))
        else:
            raise ValueError

        if self.model_cfg['rotated']:
            self.surr_model = RandomDirectionRandomForestClassifier(
                cfg, speedup=True, dim=self.dataset.D[0]
                )
        else:
            self.surr_model = RandomForestClassifier(cfg)

    def update_surrogate(self):

        self.random_init_model()

        self.surr_model.fit(*self.dataset.total_observed)

    def acquire(self):

        self.update_surrogate()

        return super().acquire()


class RandomRandomForestClassifierSurrogateAcquisitionEntropy(
        _RandomRandomForestSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


def entropy_loss(
        remaining_data, remaining_idxs, model, surr_model=None,
        eps=1e-15, T=None, cfg=None):

    model_pred = model.predict(remaining_data, idxs=remaining_idxs)

    if T is not None:
        model_pred = np.exp(np.log(model_pred)/T)

        model_pred = np.clip(model_pred, eps, 1/eps)
        model_pred[np.isnan(model_pred)] = 1/eps

        model_pred /= model_pred.sum(axis=1, keepdims=True)

        model_pred = np.clip(model_pred, eps, 1/eps)
        model_pred[np.isnan(model_pred)] = 1/eps

    if surr_model is not None:
        surr_model_pred = surr_model.predict(
            remaining_data, idxs=remaining_idxs)

        if T is not None:
            surr_model_pred = np.exp(np.log(surr_model_pred)/T)
            surr_model_pred = np.clip(surr_model_pred, eps, 1/eps)
            surr_model_pred[np.isnan(surr_model_pred)] = 1/eps

            surr_model_pred /= surr_model_pred.sum(axis=1, keepdims=True)
            surr_model_pred = np.clip(surr_model_pred, eps, 1/eps)
            surr_model_pred[np.isnan(surr_model_pred)] = 1/eps

    else:
        surr_model_pred = model_pred

    if T is None:
        model_pred = np.clip(model_pred, eps, 1 - eps)
        model_pred /= model_pred.sum(axis=1, keepdims=True)

    # Sum_{y=c} p_surr(y=c|x) log p_model(y=c|x)
    res = -1 * (surr_model_pred * np.log(model_pred)).sum(-1)

    if T is not None:
        res[np.isnan(res)] = np.nanmax(res)

    # Entropy may have zero support over some of the remaining items!
    # This is not good! Model is overconfident! Condition of estimator
    # do no longer hold!

    # clip at lowest 10 percentile of prediction (add safeguard for 0 preds)
    # clip_val = max(np.percentile(res, 10), 1e-3)
    # 1e-3 is a lot for large remaining_data, probably better as
    # 1/(100*len(remaining_data))

    if cfg is not None and not cfg.get('uniform_clip', False):
        clip_val = np.percentile(res, 10)
        res = np.clip(res, clip_val, 1/eps)

    # clipping has moved to after acquisition
    return res

def accuracy_loss(
        remaining_data, remaining_idxs, model, surr_model=None):
    # we need higher values = higher loss
    # so we will return 1 - accuracy

    model_pred = model.predict(remaining_data, idxs=remaining_idxs)

    if surr_model is not None:
        surr_model_pred = surr_model.predict(
            remaining_data, idxs=remaining_idxs)
    else:
        surr_model_pred = model_pred

    pred_classes = np.argmax(model_pred, axis=1)

    # instead of 0,1 loss we get p_surr(y|x) for accuracy

    res = 1 - surr_model_pred[np.arange(len(surr_model_pred)), pred_classes]

    res = np.maximum(res, np.max(res)*0.05)

    return res


