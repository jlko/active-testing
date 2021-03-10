import numpy as np

from activetesting.utils import maps

DEBUG_WEIGHTS = False


class RiskEstimator:
    def __init__(self, loss):
        self.loss = maps.loss[loss]()
        self.risks = np.array([[]])

    def return_and_save(self, loss):
        self.risks = np.append(self.risks, loss)
        return loss


class TrueRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model):
        super().__init__(loss)

        idxs = dataset.test_idxs
        y_true = dataset.y[idxs]
        y_pred = model.predict(dataset.x[idxs], idxs=idxs)
        self.true_loss_vals = self.loss(y_pred, y_true)
        self.true_loss = self.true_loss_vals.mean()

        self.true_loss_all_idxs = np.zeros(dataset.N)
        self.true_loss_all_idxs[idxs] = self.true_loss_vals
        # print('true loss debug', self.true_loss)

    def estimate(self, *args):
        return self.return_and_save(self.true_loss)


class TrueUnseenRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model):
        super().__init__(loss)

        # not compatible with lazy prediction
        idxs = dataset.test_unseen_idxs
        y_true = dataset.y[idxs]
        y_pred = model.predict(dataset.x[idxs], idxs=idxs)
        self.true_loss_vals = self.loss(y_pred, y_true)
        self.true_loss = self.true_loss_vals.mean()

        self.true_loss_all_idxs = np.zeros(dataset.N)
        self.true_loss_all_idxs[idxs] = self.true_loss_vals
        # print('true loss debug', self.true_loss)

    def estimate(self, *args):
        return self.return_and_save(self.true_loss)


class BiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, *args):
        super().__init__(loss)

    def estimate(self, predictions, observed, *args):
        l_i = self.loss(predictions, observed).mean()
        # print('debug', l_i)
        return self.return_and_save(l_i)


class ImportanceWeightedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        M = len(predictions)

        R = 1/M * (1/acq_weights * l_i).sum()

        return self.return_and_save(R)


class NaiveUnbiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)
        m = np.arange(1, M+1)

        v = 1/(N * acq_weights) + (M-m) / N

        R = 1/M * (v * l_i).sum()

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )
        else:
            v = 1

        R = 1/M * (v * l_i).sum()

        if DEBUG_WEIGHTS:
            if isinstance(v, int):
                v = [v]
            with open('weights.csv', 'a') as f:
                data = str(list(v)).replace('[', '').replace(']', '')
                f.write(f'{len(v)}, {data}\n')

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimatorCut(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )

            # no single weight should be more than 25 percent of all weights
            v_sum = v.sum()
            cut = 0.05
            # select those weights and cut them
            v[v > cut * v_sum] = cut * v_sum
        else:
            v = 1

        R = 1/M * (v * l_i).sum()

        if DEBUG_WEIGHTS:
            if isinstance(v, int):
                v = [v]
            with open('weights_cut25.csv', 'a') as f:
                data = str(list(v)).replace('[', '').replace(']', '')
                f.write(f'{len(v)}, {data}\n')

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimatorCut1(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )

            # no single weight should be more than 10 percent of all weights
            v_sum = v.sum()
            cut = 0.1
            # select those weights and cut them
            v[v > cut * v_sum] = 0
        else:
            v = 1

        R = 1/M * (v * l_i).sum()

        if DEBUG_WEIGHTS:
            if isinstance(v, int):
                v = [v]
            with open('weights_cut10.csv', 'a') as f:
                data = str(list(v)).replace('[', '').replace(']', '')
                f.write(f'{len(v)}, {data}\n')

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimatorCut2(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )

            # no single weight should be more than 10 percent of all weights
            v_sum = v.sum()
            cut = 0.5
            # select those weights and cut them
            v[v > cut * v_sum] = 0
        else:
            v = 1

        R = 1/M * (v * l_i).sum()

        if DEBUG_WEIGHTS:
            if isinstance(v, int):
                v = [v]
            with open('weights_cut40.csv', 'a') as f:
                data = str(list(v)).replace('[', '').replace(']', '')
                f.write(f'{len(v)}, {data}\n')

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimatorCut3(RiskEstimator):
    def __init__(self, loss, dataset, *args):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )

            # no single weight should be more than 10 percent of all weights
            v_sum = v.sum()
            cut = 0.3
            # select those weights and cut them
            v[v > cut * v_sum] = 0
        else:
            v = 1

        R = 1/M * (v * l_i).sum()

        if DEBUG_WEIGHTS:
            if isinstance(v, int):
                v = [v]
            with open('weights_cut30.csv', 'a') as f:
                data = str(list(v)).replace('[', '').replace(']', '')
                f.write(f'{len(v)}, {data}\n')

        return self.return_and_save(R)
