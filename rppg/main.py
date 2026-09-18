"""Minimal rPPG training entry for the ICASSP 2027 statistical-inference paper.

Reproduces the motivating case only: RhythmMamba_fusion_scan with the
``none`` and ``osc_real`` modulation modes (the oscillatory state-transition
variant). Only the PURE / UBFC-rPPG / MMPD datasets are wired up, which are the
three datasets used in the paper.

Deterministic training: cudnn.deterministic=True, fixed seeds (42/100/202), and
isolated DataLoader generators (train vs. valid/test), matching the protocol
described in Sec. 2 of the paper.
"""
import argparse
import os
import random

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from torch.utils.data import DataLoader

RANDOM_SEED = 100


def set_seed(seed):
    """Set global random seed + deterministic flags + DataLoader generators."""
    global RANDOM_SEED, general_generator, train_generator
    RANDOM_SEED = seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    general_generator = torch.Generator()
    general_generator.manual_seed(RANDOM_SEED)
    train_generator = torch.Generator()
    train_generator.manual_seed(RANDOM_SEED)


set_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml",
                        type=str, help="Path to the training config YAML.")
    parser.add_argument('--seed', default=100, type=int,
                        help="Random seed (default 100, backward-compatible).")
    return parser


_SUPPORTED_DATASETS = {"PURE": data_loader.PURELoader.PURELoader,
                       "UBFC": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
                       "MMPD": data_loader.MMPDLoader.MMPDLoader}


def _get_loader(dataset_name):
    if dataset_name not in _SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. "
                         f"Supported: PURE, UBFC, MMPD.")
    return _SUPPORTED_DATASETS[dataset_name]


def train_and_test(config, data_loader_dict):
    """Train and test the (single) model used in the paper."""
    if config.MODEL.NAME != "RhythmMamba_fusion_scan":
        raise ValueError("This release only supports RhythmMamba_fusion_scan.")
    model_trainer = trainer.RhythmMambaFusionScanTrainer.RhythmMambaFusionScanTrainer(
        config, data_loader_dict)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    if config.MODEL.NAME != "RhythmMamba_fusion_scan":
        raise ValueError("This release only supports RhythmMamba_fusion_scan.")
    model_trainer = trainer.RhythmMambaFusionScanTrainer.RhythmMambaFusionScanTrainer(
        config, data_loader_dict)
    model_trainer.test(data_loader_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Apply the command-line seed (if provided) and rebuild the generators.
    if getattr(args, 'seed', None) is not None and args.seed != RANDOM_SEED:
        set_seed(args.seed)

    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict()

    if config.TOOLBOX_MODE == "train_and_test":
        # train
        if config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH:
            train_data_loader = _get_loader(config.TRAIN.DATA.DATASET)(
                name="train", data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader, num_workers=1,
                batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
                worker_init_fn=seed_worker, generator=train_generator)
        else:
            data_loader_dict['train'] = None

        # valid
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH
                and not config.TEST.USE_LAST_EPOCH):
            valid_data = _get_loader(config.VALID.DATA.DATASET)(
                name="valid", data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data, num_workers=1,
                batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,
                worker_init_fn=seed_worker, generator=general_generator)
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE in ("train_and_test", "only_test"):
        # test
        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.",
                  end='\n\n')
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = _get_loader(config.TEST.DATA.DATASET)(
                name="test", data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data, num_workers=1,
                batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False,
                worker_init_fn=seed_worker, generator=general_generator)
        else:
            data_loader_dict['test'] = None

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    else:
        raise ValueError("TOOLBOX_MODE only supports train_and_test or only_test.")
