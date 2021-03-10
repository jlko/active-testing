"""Run experiment."""
import pandas as pd
import numpy as np

from activetesting.utils import maps


class Experiment:
    """Orchestrates experiment.

    Main goal: Just need to call Experiment.run_experiment()
    and a model will get trained and tested.

    This trains and actively tests the models.

    Has a step() method.
    Main loop is probably externally controlled for logging purposes.
    Maybe not..
    """
    def __init__(self, run, cfg, dataset, model, acquisition, acq_config):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model

        self.risk_estimators = {
            risk_estimator: maps.risk_estimator[risk_estimator](
                self.cfg.experiment.loss,
                self.dataset,
                self.model,
                )
            for risk_estimator in self.cfg.risk_estimators}

        true_loss = self.risk_estimators['TrueRiskEstimator'].true_loss_vals

        # TODO: model_cfg is used for surr model.
        # TODO: allow to pass a different cfg here, matching the aux model
        # TODO: this is needed when aux model and model are different!

        acq_config = model.cfg if acq_config is None else acq_config

        self.acquisition = (
            maps.acquisition[acquisition](
                [self.cfg.acquisition, run],
                self.dataset,
                true_loss_vals=true_loss,
                model=self.model,
                model_cfg=acq_config))

        self.finished = False
        self.predictions = None

    def estimate_risks(self):
        """Estimate test risk."""
        pred = self.predictions
        obs = self.dataset.y[self.dataset.test_observed]

        for risk_estimator in self.risk_estimators.values():
            risk_estimator.estimate(pred, obs, self.acquisition.weights)

    def step(self, i):
        """Perform a single testing step."""

        # choose index for next observation
        test_idx, pmf_idx = self.acquisition.acquire()

        self.observe_at_idx(i, test_idx)

        return test_idx, pmf_idx

    def observe_at_idx(self, i, idx):

        # add true pmf to logging to plot loss dist
        if self.acquisition.check_save(off=1):
            true_pmf = (
                self.risk_estimators[
                    'TrueRiskEstimator'].true_loss_all_idxs[
                    self.dataset.test_remaining])
            true_pmf = (
                self.acquisition.safe_normalise(
                    true_pmf))

            self.acquisition.all_pmfs[-1]['true_pmf'] = true_pmf

        # observe point
        x, _ = self.dataset.observe(idx)

        # predict at point
        y_pred = self.model.predict(x, [idx])

        if self.predictions is None:
            self.predictions = y_pred
        else:
            self.predictions = np.concatenate([self.predictions, y_pred], 0)

        # estimate test risk
        self.estimate_risks()

        # print(
        #     x, idx, self.dataset.test_observed.size,
        #     self.dataset.test_remaining.size)

        if len(self.dataset.test_remaining) == 0:
            self.finished = True

        if lim := self.cfg.experiment.get('abort_test_after', False):
            if i > lim:
                self.finished = True

    def external_step(self, i, test_idx, pmf_idx):
        """Externally force experiment to acquire data at 'idx'. """
        # hot-patch the forced acquisition
        # (would ideally make this passable to acquire()
        # but I can't be bothered
        self.acquisition.externally_controlled = True
        self.acquisition.ext_test_idx = test_idx
        self.acquisition.ext_pmf_idx = pmf_idx

        # need to call this s.t. acquisition weights are properly written
        self.acquisition.acquire()
        self.observe_at_idx(i, test_idx)

        # make valid for next round again
        self.acquisition.externally_controlled = False
        self.acquisition.ext_test_idx = None
        self.acquisition.ext_pmf_idx = None

    def export_data(self):
        """Extract data from experiment."""
        if self.dataset.cfg.task_type == 'classification':
            preds = np.argmax(self.predictions, 1)
        else:
            preds = self.predictions

        result = dict(
            id=np.arange(0, len(self.dataset.test_observed)),
            idx=self.dataset.test_observed,
            y_preds=preds,
            y_true=self.dataset.y[self.dataset.test_observed]
        )

        result.update(
            {risk_name: risk.risks for risk_name, risk
                in self.risk_estimators.items()})

        result = pd.DataFrame.from_dict(result)

        return result, self.acquisition.all_pmfs
