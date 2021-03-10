import math
import numpy as np
import torch

from .radial_bnn import bnn_models
from .sk2torch import SK2TorchBNN
from .cnn.models import DeepModel
from .skmodels import BaseModel

# ---- Final Pytorch Wrappers ----
# Import these models. From the outside, they can be used like Sklearn models.


def modify_bnn(model, data_CHW, *args, **kwargs):

    class Sanify(model):
        """Change default behaviour of Radial BNN.

        In particular, hide sampling behaviour and special input/output
        formatting, s.t. SaneRadialBNN behaves as a normal Pytorch model.
        """
        def __init__(self, *args, **kwargs):
            self.data_CHW = data_CHW
            super().__init__(*args, **kwargs)

        def forward(self, data, n_samples, log_sum_exp=True):
            data = self.radial_bnn_forward_reshape(data, n_samples)
            out = super().forward(data)
            if log_sum_exp:
                out = torch.logsumexp(out, dim=1) - math.log(n_samples)
            return out

        def radial_bnn_forward_reshape(self, data_N_HW, n_samples):
            # expects empty channel dimension after batch dim
            data_N_C_HW = torch.unsqueeze(data_N_HW, 1)

            if self.data_CHW is None:
                data_N_C_H_W = data_N_C_HW
            else:
                # Radial BNN and RGB Data actually does not work yet
                data_N_C_H_W = data_N_C_HW.reshape(
                    list(data_N_C_HW.shape[:-1]) + list(self.data_CHW[1:]))

            # assert len(data_N_C_H_W.shape) == 4
            data_N_V_C_H_W = torch.unsqueeze(data_N_C_H_W, 1)
            data_N_V_C_H_W = data_N_V_C_H_W.expand(
                -1, n_samples, -1, -1, -1
            )
            return data_N_V_C_H_W

    return Sanify(*args, **kwargs)


class RadialBNN(SK2TorchBNN):
    def __init__(self, cfg):
        data_CHW = cfg.get('data_CHW', None)
        kwargs = dict(channels=cfg['channels'])
        model = modify_bnn(bnn_models.RadialBNN, data_CHW, **kwargs)
        self.has_mi = True
        super().__init__(model, cfg)


class TinyRadialBNN(SK2TorchBNN):
    def __init__(self, cfg):
        data_CHW = cfg.get('data_CHW', None)
        model = modify_bnn(bnn_models.TinyRadialBNN, data_CHW)
        super().__init__(model, cfg)
        self.has_mi = True


def modify_cnns(model, data_CHW, debug_mnist):

    class Sanify(model):
        """Change default behaviour of Deterministic CNNs.

        Make them ignore args, kwargs in forward pass.
        """
        def __init__(self, *args, **kwargs):
            self.data_CHW = list(data_CHW)
            self.debug_mnist = debug_mnist
            super().__init__(*args, **kwargs)

            # original model uses Crossentropy loss
            # we use NLL loss --> need to add logsoftmax layer
            self.log_softmax = torch.nn.LogSoftmax(dim=1)

        def forward(self, data, *args, **kwargs):
            N = data.shape[0]
            data = data.reshape([N]+self.data_CHW)
            if self.debug_mnist:
                data = data.repeat(1, 3, 1, 1)

            out = super().forward(data)
            out = self.log_softmax(out)

            return out

    return Sanify


class ResNet18(SK2TorchBNN):
    def __init__(self, cfg):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'resnet18')
        super().__init__(model, cfg)


class WideResNet(SK2TorchBNN):
    def __init__(self, cfg):
        model = modify_cnns(DeepModel, cfg['data_CHW'], cfg['debug_mnist'])(
            cfg['data_CHW'][-1], cfg['num_classes'], 'wideresnet')
        super().__init__(model, cfg)


class TorchEnsemble(BaseModel):
    def __init__(self, cfg, TorchModel):
        from omegaconf import OmegaConf
        n_models = cfg['n_models']
        self.models = []
        for i in range(n_models):
            # update model save path
            if cfg.get('skip_fit_debug', False):
                cfg_i = OmegaConf.merge(
                    OmegaConf.structured(cfg),
                    dict(
                        save_path=cfg.save_path.format(i),
                        skip_fit_debug=cfg.skip_fit_debug.format(i),
                    ),
                    )
            else:
                cfg_i = cfg

            model = TorchModel(cfg_i)
            self.models.append(model)

        super().__init__(cfg, None)

    def predict(self, *args, **kwargs):
        preds = []
        for model in self.models:
            pred = model.predict(*args, **kwargs)
            preds.append(pred)

        preds = np.stack(preds, 0)
        mean_preds = np.mean(preds, 0)
        return mean_preds

    def fit(self, *args, **kwargs):

        for model in self.models:
            model.fit(*args, **kwargs)


class ResNet18Ensemble(TorchEnsemble):
    def __init__(self, cfg):
        super().__init__(cfg, ResNet18)


class WideResNetEnsemble(TorchEnsemble):
    def __init__(self, cfg):
        super().__init__(cfg, WideResNet)
