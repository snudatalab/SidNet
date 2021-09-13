#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from dotmap import DotMap
from tqdm import tqdm
import numpy as np
from method.sidnet.model import SidNet
import scipy.sparse as sp
from loguru import logger


class SidNetTrainer(torch.nn.Module):
    def __init__(self, param):
        """
        Constructore of SidNetTraininer

        :param param: parameter dictionary
        """
        super(SidNetTrainer, self).__init__()

        self.param = param
        self.device = param.device
        self.in_dim = param.in_dim
        self.c = param.hyper_param.c

    def get_normalized_matrices(self, edges, num_nodes):
        """
        Normalized signed adjacency matrix

        :param edges: signed edges
        :param num_nodes: number of nodes
        :return: normalized matrices
        """
        row, col, data = edges[:, 0], edges[:, 1], edges[:, 2]
        shaping = (num_nodes, num_nodes)
        A = sp.csr_matrix((data, (row, col)), shape=shaping)
        A = A + sp.eye(num_nodes, num_nodes)
        rowsum = np.array(np.abs(A).sum(1)).astype(np.float32)
        rowsum[rowsum == 0] = 1
        r_inv = np.power(rowsum, -1).flatten()
        r_mat_inv = sp.diags(r_inv)

        snA = r_mat_inv @ A
        snA = snA.tocoo().astype(np.float32)

        pos_idx, neg_idx = snA.data > 0, snA.data < 0
        pos_row, pos_col, pos_data = snA.row[pos_idx], snA.col[pos_idx], snA.data[pos_idx]
        neg_row, neg_col, neg_data = snA.row[neg_idx], snA.col[neg_idx], snA.data[neg_idx]
        nApT = sp.csr_matrix((np.abs(pos_data), (pos_row, pos_col)), shape=shaping).T
        nAmT = sp.csr_matrix((np.abs(neg_data), (neg_row, neg_col)), shape=shaping).T

        return nApT, nAmT

    def convert_torch_sparse(self, A, shaping):
        """
        Convert sparse matrix into torch sparse matrix

        :param A: scipy spares matrix
        :param shaping: shape
        :return: torch sparse matrix
        """
        A = A.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((A.row, A.col)).astype(np.int64))
        values = torch.from_numpy(A.data)
        return torch.sparse.FloatTensor(indices, values, shaping)

    def convert_data(self, data):
        """
        Convert input data for torch
        :param data: input data
        :return: torch data
        """
        converted_data = DotMap()
        converted_data.num_nodes = data.num_nodes
        converted_data.neg_ratio = data.neg_ratio
        converted_data.H = torch.FloatTensor(data.H).to(self.device)

        # train data
        converted_data.train.edges = torch.from_numpy(data.train.X[:, 0:3]).to(self.device)
        nApT, nAmT = self.get_normalized_matrices(data.train.X, data.num_nodes)
        nApT = self.convert_torch_sparse(nApT, nApT.shape)
        nApT = (1 - self.c) * nApT
        converted_data.train.nApT = nApT.to(self.device)

        nAmT = self.convert_torch_sparse(nAmT, nAmT.shape)
        nAmT = (1 - self.c) * nAmT
        converted_data.train.nAmT = nAmT.to(self.device)

        y = np.asarray([1 if y_val > 0 else 0 for y_val in data.train.y])
        converted_data.train.y = torch.from_numpy(y).to(self.device)
        converted_data.class_weights = torch.from_numpy(data.class_weights).type(torch.float32).to(self.device)

        # test data
        converted_data.test.edges = torch.from_numpy(data.test.X[:, 0:2]).to(self.device)
        y = np.asarray([1 if y_val > 0 else 0 for y_val in data.test.y])
        converted_data.test.y = torch.from_numpy(y).to(self.device)

        return converted_data

    def train_with_hyper_param(self, data, hyper_param, epochs=100):
        """
        Train SidNet with given hyperparameters
        :param data: input data
        :param hyper_param: hyperparameters
        :param epochs: target number of epochs
        :return: trained model
        """
        self.c = hyper_param.c
        converted_data = self.convert_data(data)

        model = SidNet(hid_dims=hyper_param.hid_dims,
                       in_dim=hyper_param.in_dim,
                       device=self.device,
                       num_nodes=converted_data.num_nodes,
                       num_layers=hyper_param.num_layers,
                       num_diff_layers=hyper_param.num_diff_layers,
                       c=hyper_param.c).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyper_param.learning_rate,
                                     weight_decay=hyper_param.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.99)

        model.train()
        pbar = tqdm(range(epochs), desc='Epoch...')

        for epoch in pbar:
            optimizer.zero_grad()
            loss = model(nApT=converted_data.train.nApT,
                         nAmT=converted_data.train.nAmT,
                         X=converted_data.H,
                         edges=converted_data.train.edges,
                         y=converted_data.train.y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description('Epoch {}: {:.4} train loss'.format(epoch, loss.item()))
        pbar.close()

        # with torch.no_grad():
        #     model.eval()
        #     auc, f1_scores, loss = model.evaluate(test_edges=converted_data.test.edges,
        #                                           test_y=converted_data.test.y)
        #     logger.info('test auc: {:.4f}'.format(auc))
        #     logger.info('test f1_macro:  {:.4f}'.format(f1_scores.macro))

        return model
