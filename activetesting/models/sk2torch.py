"""Make PyTorch models work with SKLearn interface."""
import logging
from pathlib import Path
import copy
import hydra
from omegaconf import OmegaConf

import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .radial_bnn.radial_layers.loss import Elbo
from .skmodels import BaseModel


# ---- Interface between SKLearn and Pytorch ----
# Make Pytorch model behave as SKLearn model on the outside.
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, chw, transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.chw = chw

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = x.reshape(self.chw).transpose(1, 2, 0)
            x = self.transform(x)
            x = x.reshape(-1)

        return x, y

    def __len__(self):
        return len(self.data)


class SK2TorchBNN(BaseModel):
    """Interface for Pytorch Models and SKlearn Methods."""
    def __init__(self, model, cfg):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification',))

        self.cfg = cfg
        self.t_cfg = cfg['training_cfg']
        self.model = model.to(device=self.device).type(
            torch.float32)

        self.needs_reinit = True

    def predict(self, x, for_acquisition=True, mutual_info=False, *args,
                **kwargs):

        n_samples = self.t_cfg.get('variational_samples', None)

        if len(x) > self.t_cfg['batch_size']:
            loader = self.make_loader([x], train=False)
        else:
            loader = [[torch.from_numpy(x)]]

        if for_acquisition:
            self.model.eval()

        preds = []
        with torch.no_grad():
            for (data, ) in loader:
                data = data.to(self.device)
                if not mutual_info:
                    pred = self.model(data, n_samples=n_samples)
                    # model outputs log probabilities, our code does not expect
                    # this the additional exp_logging hopefully does not
                    # introduce too much error
                    pred = torch.exp(pred)

                else:
                    # N x Samples x Classes
                    out = self.model(
                        data, n_samples=n_samples, log_sum_exp=False)

                    mean_samples = torch.logsumexp(out, dim=1) - math.log(
                        n_samples)

                    entropy_average = -torch.sum(
                        mean_samples.exp() * mean_samples, dim=1)

                    average_entropy = -torch.sum(
                        out.exp() * out, dim=2).mean(1)

                    mi = entropy_average - average_entropy

                    pred = mi

                preds.append(pred)

        preds = torch.cat(preds, 0)
        return preds.detach().cpu().numpy()

    def fit(self, x, y):

        p = self.cfg['skip_fit_debug']

        if self.cfg.get('skip_fit_debug_relative', False):
            base = Path('.')
        else:
            base = Path(hydra.utils.get_original_cwd())

        if p and (loc := (base / p)).exists():
            logging.info(f'Loading model from {p}.')
            self.model.load_state_dict(
                torch.load(loc, map_location=self.device))
        else:
            if p and not (loc := (base / p)).exists():
                logging.info(f'Tried to load model, but {p} does not exist.')

            train_loader, val_loader = self.val_train_loaders(x, y)
            # from activetesting.datasets import get_CIFAR10
            # train_loader, val_loader = get_CIFAR10()
            self.model = self.train_to_convergence(
                self.model, train_loader, val_loader)

        path = Path(self.cfg.get('save_path', 'model.pth'))

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def val_train_loaders(self, x, y):

        # generate random splits for train and val
        val_size = int(self.t_cfg['validation_set_size'])
        if self.t_cfg.get('stratify_val', False):
            strata = y
        else:
            strata = None

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=val_size, stratify=strata)

        train_loader = self.make_loader([x_train, y_train])
        val_loader = self.make_loader([x_val, y_val], train=False)

        return train_loader, val_loader

    def make_loader(self, arrs, train=True):

        if train and (self.t_cfg.get('transforms', False) == 'cifar'):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(self.cfg.data_CHW[-1], padding=4),
                transforms.RandomHorizontalFlip(),
                ])

            dataset = TransformDataset(
                arrs[0], arrs[1], list(self.cfg.data_CHW), transform)
        else:
            arrs = [torch.from_numpy(arr) for arr in arrs]
            dataset = torch.utils.data.TensorDataset(*arrs)

        bs = self.cfg.get('testing_cfg', dict()).get('batch_size', False)
        if bs and not train:
            batch_size = bs

        else:
            batch_size = self.t_cfg['batch_size']

        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=train,
            batch_size=self.t_cfg['batch_size'],
            num_workers=self.t_cfg['num_workers'],
            pin_memory=self.t_cfg['pin_memory'])

        return data_loader

    # def make_loader(self, arrs, train=True):

    #     if train and (self.t_cfg.get('transforms', False) == 'cifar'):
    #         transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             ])

    #         dataset = TransformDataset(
    #             arrs[0], arrs[1], list(self.cfg.data_CHW), transform)
    #     else:
    #         arrs = [torch.from_numpy(arr) for arr in arrs]
    #         dataset = torch.utils.data.TensorDataset(*arrs)

    #     data_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         shuffle=train,
    #         batch_size=self.t_cfg['batch_size'],
    #         num_workers=self.t_cfg['num_workers'],
    #         pin_memory=self.t_cfg['pin_memory'])

    #     return data_loader

    def train_to_convergence(self, model, train_loader, val_loader):
        logging.info(
            f'Beginning training with {len(train_loader.dataset)} training '
            f'points and {len(val_loader.dataset)} validation.'
        )
        m = self.t_cfg['max_epochs']
        log_every = int(.02 * m) if m > 100 else 1
        best = np.inf
        best_model = model
        patience = 0
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        for epoch in range(self.t_cfg['max_epochs']):

            self.train(model, train_loader, optimizer)

            val_nll, val_accuracy = self.evaluate(model, val_loader)

            if epoch % log_every == 0:
                logging.info(
                    f'Epoch {epoch:0>3d} eval: Val nll: {val_nll:.4f}, '
                    f'Val Accuracy: {val_accuracy}')

            if val_nll < best:
                best = val_nll
                best_model = copy.deepcopy(model)
                patience = 0
            else:
                patience += 1

            if patience >= self.t_cfg['early_stopping_epochs']:
                logging.info(
                    f'Patience reached - stopping training. '
                    f'Best was {best}')
                break

            if scheduler is not None:
                scheduler.step()

        logging.info('Completed training for acquisition.')

        return best_model

    def train(self, model, train_loader, optimizer):
        n_samples = self.t_cfg['variational_samples']
        model.train()

        if self.t_cfg.get('model', False) == "radial_bnn":
            loss_object = Elbo(binary=False, regression=False)
            loss_object.set_model(model, train_loader.batch_size)
            loss_object.set_num_batches(len(train_loader))

            def loss_helper(pred, target):
                nll_loss, kl_loss = loss_object.compute_loss(pred, target)
                return (nll_loss + kl_loss / 10).mean(dim=0)

            raw_loss = loss_helper

        else:
            # raw_loss = F.cross_entropy
            raw_loss = F.nll_loss

        for idx, (data, target) in enumerate(train_loader):

            data = data.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()

            prediction = model(data, n_samples, log_sum_exp=False)

            loss = raw_loss(prediction, target)

            loss.backward()

            optimizer.step()

    def evaluate(self, model, eval_loader):
        self.model.train()
        # We actually only want eval mode on when we're doing acquisition
        # because of how consistent dropout works.
        n_samples = self.t_cfg['variational_samples']
        nll = correct = 0

        with torch.no_grad():
            for data, target_N in eval_loader:

                data = data.to(self.device)
                target_N = target_N.to(self.device)

                prediction_N = model(data, n_samples=n_samples)

                # if self.cfg.get('model', False) == "radial_bnn":
                #     raw_loss = F.nll_loss
                # else:
                #     raw_loss = F.cross_entropy
                raw_loss = F.nll_loss

                raw_nll_N = raw_loss(
                    prediction_N, target_N, reduction="none")

                nll += torch.sum(raw_nll_N)

                # get the index of the max log-probability
                class_prediction = prediction_N.max(1, keepdim=True)[1]
                correct += class_prediction.eq(
                    target_N.view_as(class_prediction)).sum().item()

        nll /= len(eval_loader.dataset)

        percentage_correct = 100.0 * correct / len(eval_loader.dataset)

        return nll.item(), percentage_correct

    def get_optimizer(self):
        c = self.t_cfg.get('optimizer', False)

        if not c:
            optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.t_cfg['learning_rate'],
                    weight_decay=self.t_cfg['weight_decay'],)

        elif c == 'cifar':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.t_cfg['learning_rate'],
                momentum=0.9, weight_decay=self.t_cfg['weight_decay'])
        else:
            raise ValueError

        return optimizer

    def get_scheduler(self, optimizer):
        c = self.t_cfg.get('scheduler', False)
        epochs = self.t_cfg['max_epochs']

        if not c:
            scheduler = None

        elif c == 'cifar':
            milestones = [int(epochs * 0.5), int(epochs * 0.75)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.1)

        elif c == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)

        elif c == 'devries':
            # https://arxiv.org/abs/1708.04552v2
            assert epochs == 200
            milestones = [60, 120, 160]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2)

        return scheduler
# ---- Make Radial BNN Sane ----
# There is some unexpected behaviour in terms of the data shapes
# expected as input and given as output from Radial BNN.
# Make them behave more similar to any other PyTorch model.



# If I ever want to build sane MCDO, use this to modify forward
# if model_arch == "consistent_mcdo":
#     prediction = torch.logsumexp(
#         model(data_N_C_H_W, variational_samples), dim=1
#     ) - math.log(variational_samples)