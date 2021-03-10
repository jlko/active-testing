"""Helper class for experiment visualization.
Note that the code to create the final plots for the paper is in `plotting`.
"""

from pathlib import Path
import itertools
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf


class Visualiser:

    def __init__(self, path, task_type='regression', true_unseen=False,
                 rolling=False):

        self.task_type = task_type
        self.rolling = rolling

        if isinstance(path, list):
            self.results = self.read_multiple_experiments(path)
            self.path = Path(path[0])
        else:
            self.path = Path(path)
            self.results = pd.read_pickle(self.path / 'results.pkl')

        # Read main experiment res900ults.
        try:
            self.datasets = pickle.load(open(self.path / 'datasets.pkl', 'rb'))
        except:
            self.datasets = None
            print('Found no datasets saved.')

        # backward compatibility to old code
        try:
            self.D = self.datasets[0]['x'].shape[1]
        except:
            self.D = 1
        try:
            self.cfg = OmegaConf.load(self.path / '.hydra'/ 'config.yaml')

            if isinstance(path, list):
                self.cfgs = []
                for p in path:
                    self.cfgs.append(OmegaConf.load(Path(p) / '.hydra'/ 'config.yaml'))
            else:
                self.cfgs = None
        except:
            self.cfg = None

        if (pmf_path := self.path / 'pmfs.pkl').is_file():
            self.pmfs = pickle.load(open(pmf_path, 'rb'))
        else:
            self.pmfs = None

        # Read some experiment properties.
        # TODO: Replace with live read from config or df.
        if not true_unseen:
            self.true_risk = 'TrueRiskEstimator'
        else:
            self.true_risk = 'TrueUnseenRiskEstimator'

        risks = list(filter(
            lambda x: 'RiskEstimator' in x, self.results.columns))
        risks.remove(self.true_risk)
        self.risks = risks
        self.acquisitions = self.results['acquisition'].unique()
        self.n_points = self.results[['id']].max().values[0] + 1
        self.n_runs = self.results.run.max() + 1

        # set up some properties
        self._diffs = None
        self._means = None
        self._medians = None
        self._stds = None
        self._percentiles = None
        self._errors = None
        self._quant_errors = None
        self.q = 0.5
        self._extra_quant_errors = None
        self._log_sq_diff = None
        self._log_sq_diff_std = None

    def read_multiple_experiments(self, paths):
        for i, path in enumerate(paths):
            res = pd.read_pickle(Path(path) / 'results.pkl')
            if i > 0:
                res['run'] = results['run'].max() + res['run'] + 1
                results = results.append(res, ignore_index=True)
            else:
                results = res

        return results

    def config(self):
        print(OmegaConf.to_yaml(self.cfg))

    @property
    def diffs(self):
        if self._diffs is None:
            diffs = self.results.copy()
            for risk in self.risks:
                a = self.results[risk]
                b = self.results[self.true_risk]
                diffs[risk] = a - b
            self._diffs = diffs
        return self._diffs

    @property
    def means(self):
        if self._means is None:
            self._means = self.diffs.groupby(['acquisition', 'id']).mean()
            self._means = self._means[self.risks]
        return self._means

    @property
    def medians(self):
        if self._medians is None:
            self._medians = self.diffs.groupby(['acquisition', 'id']).median()
            self._medians = self._medians[self.risks]
        return self._medians

    @property
    def stds(self):
        if self._stds is None:
            self._stds = self.diffs.groupby(['acquisition', 'id']).std()
            self._stds = self._stds[self.risks]
        return self._stds

    @property
    def percentiles(self):
        if self._percentiles is None:
            group = self.diffs.groupby(['acquisition', 'id'])
            self._percentiles = [
                group.quantile(0.1),
                group.quantile(0.9)]
        return self._percentiles

    @property
    def errors(self):
        if self._errors is None:
            self._errors = self.diffs.copy()
            for risk in self.risks:
                self._errors[risk] = self._errors[risk]**2

            self._errors = self._errors.groupby(['acquisition', 'id']).mean()
            # self._errors = np.sqrt(self._errors[self.risks])
            if self.rolling is not False:
                self._quant_errors = self._quant_errors.rolling(
                    self.rolling).mean()
        return self._errors

    @property
    def quant_errors(self, q=0.5):
        if (self._quant_errors is None) or (q != self.q):
            self.q = q
            self._quant_errors = self.diffs.copy()
            for risk in self.risks:
                self._quant_errors[risk] = self._quant_errors[risk]**2

            self._quant_errors = self._quant_errors.groupby(
                ['acquisition', 'id']).quantile(self.q)
            # self._quant_errors = np.sqrt(self._quant_errors[self.risks])
            if self.rolling is not False:
                self._quant_errors = self._quant_errors.rolling(
                    self.rolling).mean()

        return self._quant_errors

    @property
    def extra_quant_errors(self):
        if self._extra_quant_errors is None:
            self._extra_quant_errors = self.diffs.copy()
            for risk in self.risks:
                self._extra_quant_errors[risk] = self._extra_quant_errors[risk]**2
            group = self.extra_quant_errors.groupby(['acquisition', 'id'])

            self._extra_quant_errors = [
                group.quantile(0.25), group.quantile(0.75)]

        return self._extra_quant_errors

    @property
    def log_sq_diff(self):
        if self._log_sq_diff is None:
            self._log_sq_diff = self.diffs.copy()
            for risk in self.risks:
                self._log_sq_diff[risk] = np.log10(self._log_sq_diff[risk]**2)
            self._log_sq_diff = self._log_sq_diff.groupby(
                ['acquisition', 'id']).mean()
            self._log_sq_diff = self._log_sq_diff[self.risks]
        return self._log_sq_diff

    @property
    def log_sq_diff_std(self):
        if self._log_sq_diff_std is None:
            self._log_sq_diff_std = self.diffs.copy()
            for risk in self.risks:
                self._log_sq_diff_std[risk] = np.log10(self._log_sq_diff_std[risk]**2)
            self._log_sq_diff_std = self._log_sq_diff_std.groupby(
                ['acquisition', 'id']).std()
            self._log_sq_diff_std = self._log_sq_diff_std[self.risks]
        return self._log_sq_diff_std

    def get_slice(self, risk, acquisition, step):

        r = self.diffs
        cond = (r.acquisition == acquisition) & (r.id == step)

        return r[cond].sort_values('run')[risk]**2
        # return r[cond][risk]**2

    def plot_risks(self, errors='std_error', acquisitions=None, risks=None,
                   fig=None, ax=None):
        """Plot mean +- std error.

        Of the risk-estimator-acquisition-function combination against the
        true empirical test risk.

        """
        if errors == 'std_error':
            sqrtN = np.sqrt(self.n_runs)
        elif errors == 'std':
            sqrtN = 1
        else:
            raise ValueError(f'Do not recognize errors={errors}.')
        if acquisitions is None:
            acquisitions = self.acquisitions
        if risks is None:
            risks = self.risks
        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        x = np.arange(0, self.n_points)
        i = 0
        linestyles = itertools.cycle(['--', '-.', ':'])
        for risk, ls in zip(risks, linestyles):
            for acquisition in acquisitions:
                i += 1
                m = self.means.loc[acquisition][risk].values
                s = self.stds.loc[acquisition][risk].values
                ax.plot(x, m, color=f'C{i}', linestyle=ls,
                        label=f'{risk}_{acquisition}')
                ax.fill_between(
                    x, m+s/sqrtN, m-s/sqrtN,
                    color=f'C{i}', alpha=0.3)

        plt.legend(fontsize=6)

        return fig, ax

    def plot_risks_select_combinations(
                self, acquisition_risks, errors='std',
                fig=None, ax=None, alpha=0.3, i=0):
        """Plot mean +- std error.

        Of the risk-estimator-acquisition-function combination against the
        true empirical test risk.

        """
        if errors == 'std_error':
            middle = self.means
            sqrtN = np.sqrt(self.n_runs)
            upper_base = middle + self.stds/sqrtN
            lower_base = middle - self.stds/sqrtN
        elif errors == 'std':
            middle = self.means
            upper_base = middle + self.stds
            lower_base = middle - self.stds
        elif errors == 'percentiles':
            middle = self.means
            lower_base, upper_base = self.percentiles
        else:
            raise ValueError(f'Do not recognize errors={errors}.')
        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        x = np.arange(1, self.n_points+1)
        linestyles = itertools.cycle(['--', '-.', ':'])
        for acquisition, risk in acquisition_risks:
            i += 1
            m = middle.loc[acquisition][risk].values
            s_u = upper_base.loc[acquisition][risk].values
            s_l = lower_base.loc[acquisition][risk].values
            ax.fill_between(
                x, s_u, s_l,
                color=f'C{i}', alpha=alpha)
            ax.plot(x, s_l, '--', color=f'C{i}', zorder=100)
            ax.plot(x, s_u, '--', color=f'C{i}', zorder=100)
            ax.plot(x, m, color=f'C{i}',
                    label=f'{risk}_{acquisition}'.replace('_', '-'),
                    zorder=100)

        plt.legend(fontsize=6)

        return fig, ax

    def plot_log_convergence(
                self, acquisition_risks, errors='std',
                fig=None, ax=None, alpha=0.3, i=0, rolling=False,
                label_add=''):
        """Plot mean +- std error.

        Of the risk-estimator-acquisition-function combination against the
        true empirical test risk.

        """
        if errors == 'percentiles':
            upper_base = self.quant_errors
        elif errors == 'std':
            upper_base = self.errors
        else:
            raise ValueError

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        x = np.arange(0, self.n_points)
        linestyles = itertools.cycle(['--', '-.', ':'])
        for acquisition, risk in acquisition_risks:
            i += 1
            s_u = upper_base.loc[acquisition][risk].values
            if (R := rolling) is not False:
                s_u = np.convolve(
                    s_u, np.ones(R)/R, mode='valid')
                x = np.arange(0, len(s_u))
            ax.loglog(
                x, s_u, '-', color=f'C{i}',
                label=f'{risk}-{acquisition}{label_add}'.replace('_', '-'),
                zorder=100)

        plt.legend(fontsize=6)

        return fig, ax

    def plot_ratios(
                self, acquisition_risks, errors='std',
                fig=None, ax=None, alpha=0.3, i=0, smoothing=False):
        """Plot mean +- std error.

        Of the risk-estimator-acquisition-function combination against the
        true empirical test risk.

        """
        if errors == 'percentiles':
            upper_base = self.quant_errors
        elif errors == 'std':
            upper_base = self.errors
        else:
            raise ValueError

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        x = np.arange(1, self.n_points+1)
        linestyles = itertools.cycle(['--', '-.', ':'])
        for acquisition, risk in acquisition_risks:
            i += 1

            selected = upper_base.loc[
                'RandomAcquisition']['BiasedRiskEstimator'].values
            random = upper_base.loc[acquisition][risk].values

            y = selected / random

            ax.plot(
                x, y, '-', color=f'C{i}', label=f'{risk}-{acquisition}'.replace('_', '-'),
                zorder=100)

            # ax.semilogx(np.log10(x[1:]), np.diff(np.log10(s_u)), label=f'{risk}_{acquisition}',
            #     zorder=100)

        plt.legend(fontsize=6)

        return fig, ax

    def plot_bars_from_ratios(
                self, acquisition_risks, errors='std',
                fig=None, ax=None, alpha=0.3, i=0, smoothing=False,
                x_min_max=None
                ):
        """Plot mean +- std error.

        Of the risk-estimator-acquisition-function combination against the
        true empirical test risk.

        """
        if errors == 'percentiles':
            upper_base = self.quant_errors
        elif errors == 'std':
            upper_base = self.errors
        else:
            raise ValueError

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        x = np.arange(1, self.n_points+1)
        linestyles = itertools.cycle(['--', '-.', ':'])
        i = 1
        for acquisition, risk in acquisition_risks:
            i += 1

            selected = upper_base.loc[
                'RandomAcquisition']['BiasedRiskEstimator'].values
            random = upper_base.loc[acquisition][risk].values

            ratios = selected / random

            if (a := x_min_max) is not None:
                ratios = ratios[a[0]:a[1]]

            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            print(acquisition, mean_ratio, std_ratio)
            ax.bar(
                i, mean_ratio, color=f'C{i}', label=f'{risk}-{acquisition}'.replace('_', '-'),
                alpha=0.7
                )
            ax.plot(
                [i, i], [mean_ratio+std_ratio, mean_ratio-std_ratio],
                color='grey', lw=2
            )

        # ax.plot([1, i+0.5], [1, 1], c=f'C1', label='random')

        plt.legend(fontsize=6)

        return fig, ax

    def plot_equivalent_samples(
                self, acquisition_risks, baseline=None, errors='std',
                fig=None, ax=None, alpha=0.3, i=0, relative=True,
                rolling_before=False, rolling_after=False, inverse=True,
                ):
        """Plot mean +- std error.

        Of the risk-estimator-acquisition-function combination against the
        true empirical test risk.

        """
        if errors == 'percentiles':
            upper_base = self.quant_errors
        elif errors == 'std':
            upper_base = self.errors
        else:
            raise ValueError

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        if baseline is None:
            baselines = ['RandomAcquisition', 'BiasedRiskEstimator']

        base_risk = upper_base.loc[baselines[0]][baselines[1]].values

        x = np.arange(0, self.n_points)
        linestyles = itertools.cycle(['--', '-.', ':'])
        for acquisition, risk in acquisition_risks:
            i += 1
            s_u = upper_base.loc[acquisition][risk].values
            if (R := rolling_before) is not False:
                s_u = np.convolve(
                    s_u, np.ones(R)/R, mode='valid')
                base_risk = np.convolve(
                    base_risk, np.ones(R)/R, mode='valid')

            diffs = s_u[:, np.newaxis] - base_risk
            diffs[diffs < 0] = 1e10
            idxs = np.argmin(diffs, axis=1) + 1
            x = range(1, len(idxs)+1)
            if relative:
                y = idxs/x
            else:
                y = idxs

            if (R := rolling_after) is not False:
                y = np.convolve(y, np.ones(R)/R, mode='valid')
                x = range(1, len(y)+1)

            if inverse:
                y = 1/y

            ax.plot(
                x, y, '-', color=f'C{i}',
                label=f'{risk}_{acquisition}'.replace('_', '-'),
                zorder=100)

        ax.plot(x, len(x)*[1], '--', c='grey', alpha=0.5, zorder=-1, label='Above: Better, Below: Worse')

        plt.legend(fontsize=6)
        ax.set_xlabel('Active Test Samples')
        ax.set_ylabel('Efficiency Improvement Factor')
        plt.grid()

        return fig, ax

    def plot_data(self, run=0, fig=None, ax=None, legend=True, paper=False):

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        if self.task_type == 'regression' and self.D == 1:
            return self.plot_1D_regression_data(run, fig, ax, legend)
        elif self.task_type == 'classification' and self.D == 2:
            return self.plot_2D_classification_data(
                run, fig, ax, legend, paper)
        else:
            raise ValueError

    def plot_1D_regression_data(self, run, fig, ax, legend):

        data = self.datasets[run]
        data['x'] = self.check_remove_feature_dim(data['x'])

        train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]
        test = data['x'][data['test_idxs']], data['y'][data['test_idxs']]

        # also plot model predictions
        filtered = self.results[
            (self.results.run == run)
            & (self.results.acquisition == self.acquisitions[0])]
        y_preds, idx = filtered.y_preds.values, filtered.idx.values
        sorted_idxs = np.argsort(data['x'][idx])
        predicted_p = ax.plot(
            data['x'][idx][sorted_idxs], y_preds[sorted_idxs],
            zorder=-10, linewidth=1, color='grey', alpha=1, label='model')

        ax.scatter(*train, s=1, label='train')
        ax.scatter(*test, s=1, label='test')
        if legend:
            ax.legend(fontsize=6)

        return fig, ax

    def plot_2D_classification_data(self, run, fig, ax, legend, paper):
        data = self.datasets[run]

        train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]
        test = data['x'][data['test_idxs']], data['y'][data['test_idxs']]

        # also plot model predictions
        filtered = self.results[
            (self.results.run == run)
            & (self.results.acquisition == self.acquisitions[0])]
        y_preds, idx = filtered.y_preds.values, filtered.idx.values

        ax.scatter(
            data['x'][idx][:, 0],
            data['x'][idx][:, 1],
            c=y_preds,
            label='model',
            s=200,
            marker='$\mathrm{o}$',
            alpha=0.5,
            )

        ax.scatter(
            train[0][:, 0], train[0][:, 1], c=train[1],
            s=10, marker='D', label='train')

        ax.scatter(
            test[0][:, 0], test[0][:, 1], c=test[1],
            s=10, marker='x', label='test')
        if legend:
            ax.legend(fontsize=6)

        return fig, ax

    def animate_acquisition(self, acquisition, run=0):
        from celluloid import Camera

        if self.pmfs is None:
            raise ValueError(f'No pmfs are loaded!')

        pmfs = self.pmfs[run][acquisition]
        data = self.datasets[run]
        data['x'] = self.check_remove_feature_dim(data['x'])

        train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]

        filtered = self.results[
            (self.results.run == run)
            & (self.results.acquisition == acquisition)]
        y_preds, idx = filtered.y_preds.values, filtered.idx.values

        fig, ax = plt.subplots(2, dpi=200)
        camera = Camera(fig)

        # fig.suptitle(f'Acquisition: {acquisition} for Run {run}')
        N = len(data['x'])
        for t, pmf in enumerate(pmfs):

            test_remain = (
                data['x'][pmf['remaining']], data['y'][pmf['remaining']])
            test_select = (
                data['x'][pmf['observed']], data['y'][pmf['observed']])

            # plot train and test data
            train_p = ax[0].plot(*train, '.', markersize=1, c='grey')
            remaining_p = ax[0].plot(*test_remain, 'o', markersize=1, c='C2')
            selected_p = ax[0].plot(*test_select, 'o', markersize=1, c='C6')
            # current choice
            current_p = ax[0].plot(
                data['x'][pmf['test_idx']], data['y'][pmf['test_idx']],
                '.', color='r', markersize=5)

            # also plot static predictions of model
            sorted_idxs = np.argsort(data['x'][idx])
            predicted_p = ax[0].plot(
                data['x'][idx][sorted_idxs], y_preds[sorted_idxs],
                zorder=-1, linewidth=0.5, color='grey', alpha=0.4)

            # now plot acquisition distributions
            align_pmf = np.zeros(N)
            align_pmf[pmf['remaining']] = pmf['pmf']
            align_pmf /= align_pmf.max()

            bar_p = ax[1].bar(np.arange(N), align_pmf, color='C0')
            # current choice
            bar_select_p = ax[1].bar(
                pmf['test_idx'], align_pmf[pmf['test_idx']],
                color='r', width=1.2)

            # legend
            # unfortunately, this has to be done in this way
            # to not break compatibility with celluloid
            ax[0].legend(
                (list(train_p)+list(remaining_p)+list(selected_p)
                 + list(predicted_p)+list(current_p)),
                ['Train Data', 'Test Remaining', 'Test Observed',
                 'Model Predictions', 'Current Point'],
                fontsize=6, loc='upper left')
            ax[1].legend(
                list(bar_p),
                ['Acquisition Function'],
                fontsize=6, loc='upper left')

            camera.snap()
        animation = camera.animate()
        save_path = self.path / f'animation_run{run:02d}_{acquisition}.gif'
        animation.save(save_path)
        print(f'Saved to {save_path}.')

        return fig, ax

    @staticmethod
    def check_remove_feature_dim(arr):
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr

    def plot_all_runs(self, acquisition, risk, break_after=10000, runs=None):
        # There may be crazy outliers. Probably due to standardisation
        tmp = self.diffs[self.diffs.acquisition == acquisition]

        fig, ax = plt.subplots(1, 1, dpi=200)

        runs = sorted(self.diffs.run.unique()) if runs is None else runs

        for run in runs:
            run_data = tmp[tmp.run == run]
            run_data.plot('id', risk, ax=ax, label='')

            if run > break_after:
                break

        return fig, ax

    def plot_outliers(self, acquisition, risk, break_after=10000, runs=None,
                      omit=[0, 1e19]):
        # There may be crazy outliers. Probably due to standardisation
        tmp = self.diffs[self.diffs.acquisition == acquisition]

        fig, ax = plt.subplots(1, 1, dpi=200)

        print('sorting')
        runs = sorted(self.diffs.run.unique()) if runs is None else runs
        print('plitting')

        omitted = []

        for run in runs:
            run_data = tmp[tmp.run == run]
            x = run_data.id.values
            y = np.abs(run_data[risk].values)

            if np.max(y[omit[0]:]) > omit[1]:
                omitted.append(run)
                continue

            plt.plot(x, y)

            if run > break_after:
                break

        print(f'Ommitted {len(omitted)} runs: {omitted}.')

        return fig, ax, omitted

    def loss_dist(self, acquisition, run=0, step=0, fig=None, ax=None,
                  normalise=False, T=None, lazy_save=False):

        if self.pmfs is None:
            raise ValueError(f'No pmfs are loaded!')

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=200)

        vals = self.pmfs[run][acquisition]
        pmfs = vals[step]
        acq = pmfs['pmf']
        # true = pmfs['true_pmf']
        true = self.pmfs[run]['TrueLossAcquisition'][step]['pmf']

        acq, true = np.array(acq), np.array(true)

        # Normalise to max val s.t. celluloid can keep constant axes.
        max_val = np.max([acq.max(), true.max()])
        min_val = np.min([acq.min(), true.min()])
        if normalise:
            acq = (acq-acq.min())/(acq.max() - acq.min())
            true = (true-true.min())/(true.max()-true.min())
            max_val = 1
            min_val = 0

        sorted_idxs = np.argsort(true)

        true_p = ax.scatter(
            np.arange(len(true[sorted_idxs])),
            true[sorted_idxs], label='true', c='C0', s=0.5)

        acq_p = ax.scatter(
            np.arange(len(acq[sorted_idxs])),
            acq[sorted_idxs], label='acquisition', c='C1', s=1)

        # This is broken for lazy evaluation
        if not lazy_save:
            # Also plot the one that is currently being selected
            # set difference between remaining at this step and remaining at next step
            # --> this one get's sampled
            remaining = vals[step]['remaining']
            # index in original data
            if len(remaining) > 1:
                data_idx = list(set(remaining)-set(vals[step+1]['remaining']))[0]
            else:
                data_idx = remaining[0]
            # where is this in the unsorted array
            remaining_idx = np.where(remaining == data_idx)[0][0]
            # where does this get mapped to
            # this is finally where the acquired point sits
            sorted_idx = np.where(sorted_idxs == remaining_idx)[0][0]

            acquired = ax.plot(
                [sorted_idx, sorted_idx], [min_val, max_val], '--', c='grey')

        # ax.legend(
        #     (list(true_p)+list(acq_p)+list(acquired)),
        #     ['True Loss', 'Acquisition', 'Acquired'],
        #     fontsize=6, loc='upper left')

        return fig, ax

    def animate_loss_dist(self, acquisition, run=0):
        from celluloid import Camera

        fig, ax = plt.subplots(dpi=200)
        camera = Camera(fig)

        fig.suptitle(f'Acquisition: {acquisition} for run {run}')

        for step in range(self.n_points):

            self.loss_dist(
                acquisition, run=run, step=step, fig=fig, ax=ax,
                normalise=True, lazy_save=False)
            camera.snap()

        animation = camera.animate()
        save_path = self.path / f'animation_loss_dist{run:02d}_{acquisition}.gif'
        animation.save(save_path)
        print(f'Saved to {save_path}.')

        return fig, ax


class MultiVisualiser:
    def __init__(self, path, *args, **kwargs):
        self.visualisers = dict()
        self.path = path
        for p in path.glob('model_*'):
            if '.' in str(p):
                continue
            self.visualisers[p.stem] = Visualiser(p, *args, **kwargs)

        self.cfg = OmegaConf.load(self.path / '.hydra' / 'config.yaml')

    def __iter__(self):
        for name, vis in self.visualisers.items():
            yield name, vis

    def config(self):
        print(OmegaConf.to_yaml(self.cfg))

    def __getitem__(self, name):
        return self.visualisers[name]

    def keys(self):
        return self.visualisers.keys()

    def values(self):
        return self.visualisers.values()
    
    def items(self):
        return self.visualisers.items()
