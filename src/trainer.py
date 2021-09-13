#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import json

import os
import fire
import sys
import torch
from loguru import logger
from dotmap import DotMap
import utils

from data.data_loader import DataLoader
from method.sidnet.train import SidNetTrainer


def make_dirs(path):
    """
    Make directories
    :param path: target path
    """
    os.system('mkdir -p {}'.format(path))


def run(param):
    """
    Train SidNet
    :param param: parameters
    """
    data_loader = DataLoader(random_seed=param.random_seed,
                             reduction_dimension=param.reduction_dimension,
                             reduction_iterations=param.reduction_iterations)

    # data = {train, test}, train = {X, y}, test = {X, y} according to heldout_ratio
    data = data_loader.load(data_path=param.data_path,
                            heldout_ratio=param.heldout_ratio)

    logger.info('Start training SidNet with the hyperparameters...')

    # training
    trainer = SidNetTrainer(param)
    trained_model = trainer.train_with_hyper_param(data=data,
                                                   hyper_param=param.hyper_param,
                                                   epochs=param.epochs)

    # save model
    logger.info('Save the trained model at {}...'.format(param.output_home))
    logger.info('- Path for the trained model: {}'.format(param.paths.model_output_path))
    logger.info('- Path for the hyperparameters used in the model: {}'.format(param.paths.param_output_path))

    torch.save(trained_model.state_dict(), param.paths.model_output_path)
    param.device = 0
    with open(param.paths.param_output_path, 'w') as out_file:
        json.dump(param, out_file)


def main(data_home='../data',
         output_home='../output',
         dataset='BITCOIN_ALPHA',
         heldout_ratio=0.2,
         random_seed=1,
         use_torch_random_seed=True,
         reduction_dimension=128,
         reduction_iterations=30,
         gpu_id=0,
         c=0.15,
         weight_decay=1e-3,
         learning_rate=0.01,
         num_layers=1,
         hid_dim=32,
         num_diff_layers=10,
         epochs=100):
    """
    Handle user arguments

    :param data_home: home directory path for data
    :param output_home: output directory path
    :param dataset: dataset name
    :param heldout_ratio: heldout ratio between training and test
    :param random_seed: random seed
    :param reduction_dimension: input feature dimension (SVD)
    :param reduction_iterations: number of iterations required by SVD computation
    :param gpu_id: gpu id
    :param weight_decay: weight decay (L2 regularizer) for optimizer
    :param learning_rate: learning rate for optimizer
    :param c: ratio of local feature injection
    :param num_layers: number of layers (L)
    :param hid_dim: hidden feature dimension (d)
    :param num_diff_layers: number of diffusion steps (K)
    :param epochs: target number of epochs
    """

    device = torch.device(f"cuda:{gpu_id}"
                          if (torch.cuda.is_available() and gpu_id >= 0)
                          else "cpu")

    torch.manual_seed(random_seed)

    param = DotMap()

    # gpu
    param.device = device

    # data parameters
    param.data_home = data_home
    param.data_path = f'{data_home}/{dataset}/data.tsv'
    param.dataset = dataset
    param.output_home = output_home
    param.heldout_ratio = heldout_ratio
    #param.use_random_seed = use_random_seed
    param.random_seed = random_seed
    param.run_id = random_seed
    param.reduction_dimension = reduction_dimension
    param.reduction_iterations = reduction_iterations

    param.gpu_id = gpu_id

    # hyperparameters
    hyper_param = DotMap()
    hyper_param.num_layers = num_layers
    hyper_param.num_diff_layers = num_diff_layers
    hyper_param.c = c
    hyper_param.in_dim = reduction_dimension
    hyper_param.hid_dim = hid_dim
    hyper_param.hid_dims = [hid_dim] * (num_layers)
    hyper_param.weight_decay = weight_decay
    hyper_param.learning_rate = learning_rate
    param.hyper_param = hyper_param

    # method parameters
    param.in_dim = reduction_dimension

    # optimizer
    param.epochs = epochs

    # paths
    paths = DotMap()
    paths.test_dir = f'{output_home}/{dataset}'
    for (key, path) in paths.items():
        make_dirs(path)

    paths.model_output_path = f'{output_home}/{dataset}/model.pt'
    paths.param_output_path = f'{output_home}/{dataset}/param.json'
    param.paths = paths

    # torch seed
    param.use_torch_random_seed = use_torch_random_seed
    if use_torch_random_seed:
        param.torch_seed = torch.initial_seed()

    utils.log_param(param)

    run(param)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))