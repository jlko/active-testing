"""Main active testing loop."""
import os
import logging
import hydra
import warnings

import numpy as np
import torch

from activetesting.experiment import Experiment
from activetesting.utils import maps
from activetesting.hoover import Hoover
from activetesting.models import make_efficient
from omegaconf import OmegaConf


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    """Run main experiment loop.

    Repeat active testing across multiple data splits and acquisition
    functions for all risk estimators.
    """

    rng = cfg.experiment.random_seed
    if rng == -1:
        rng = np.random.randint(0, 1000)

    if rng is not False:
        np.random.seed(rng)
        torch.torch.manual_seed(rng)
    logging.info(f'Setting random seed to {rng}.')

    hoover = Hoover(cfg.hoover)
    logging.info(f'Logging to {os.getcwd()}.')

    model = None

    # Right now this averages over both train and testing!
    for run in range(cfg.experiment.n_runs):
        if run % cfg.experiment.log_every == 0 or cfg.experiment.debug:
            logging.info(f'Run {run} in {os.getcwd()}.')
            if cuda := torch.cuda.is_available():
                logging.info(f'Still using cuda: {cuda}.')
            else:
                os.system('touch cuda_failure.txt')

        dataset = maps.dataset[cfg.dataset.name](cfg.dataset, cfg.model)

        # Train model on training data.
        if (not cfg.model.get('keep_constant', False)) or (model is None):
            # default case
            model = maps.model[cfg.model.name](cfg.model)
            model.fit(*dataset.train_data)

        # Always predict on test data again
        # TODO: need to fix this for efficient prediction
        if cfg.model.get('efficient', False):
            logging.info('Eficient prediction on test set.')
            model = make_efficient(model, dataset)

        # if cfg.experiment.debug:
            # Report train error
            # logging.info('Model train error:')
            # model.performance(
            #     *dataset.train_data, dataset.cfg.task_type)

        # if not check_valid(model, dataset):
            # continue

        if run < cfg.experiment.save_data_until:
            hoover.add_data(run, dataset.export())

        for acq_dict in cfg.acquisition_functions:
            # Slightly unclean, but could not figure out how to make
            # this work with Hydra otherwise
            acquisition = list(acq_dict.keys())[0]
            acq_cfg_name = list(acq_dict.values())[0]

            if cfg.experiment.debug:
                logging.info(f'\t Acquisition: {acquisition}')

            if (n := acq_cfg_name) is not None:
                acq_config = cfg['acquisition_configs'][n]
            else:
                acq_config = None

            experiment = Experiment(
                run, cfg, dataset, model, acquisition, acq_config)

            i = 0
            while not experiment.finished:
                i += 1
                if cfg.experiment.debug:
                    logging.info(
                        f'\t Acquisition: {acquisition} – \t Step {i}.')

                experiment.step(i)

            # Add config to name for logging.
            if (n := acq_cfg_name) is not None:
                acquisition = f'{acquisition}_{n}'

            # Extract results from acquisition experiment
            hoover.add_results(run, acquisition, experiment.export_data())

            # Reset selected test_indices.
            dataset.restart()

        if run % cfg.experiment.get('save_every', 1e19) == 0:
            logging.info('Intermediate save.')
            hoover.save()

    logging.info('Completed all runs.')
    hoover.save()


def check_valid(model, dataset):
    """For classification with small number of points and unstratified."""
    if hasattr(model.model, 'n_classes_'):
        if (nc := model.model.n_classes_) != dataset.cfg.n_classes:
            warnings.warn(
                f'Not all classes present in train data. '
                f'Skipping run.')
            return False
    return True


if __name__ == '__main__':
    import os
    os.environ['HYDRA_FULL_ERROR'] = '1'

    def get_base_dir():
        return os.getenv('BASE_DIR', default='.')

    OmegaConf.register_resolver('BASE_DIR', get_base_dir)

    main()