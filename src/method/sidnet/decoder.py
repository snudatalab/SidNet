#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_accuracies


class Decoder(nn.Module):
    def __init__(self, emb_size):
        """
        Constructor of Decoder
        :param emb_size: embedding size
        """
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.W = nn.Parameter(torch.FloatTensor(2 * self.emb_size, 2))
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, Z, edges, y):
        """
        Forward edge features from Z into binary cross entropy loss

        :param Z: final node embeddings
        :param edges: edges
        :param y: signs
        :return: binary cross entropy loss
        """
        src_features = Z[edges[:, 0], :]
        dst_features = Z[edges[:, 1], :]
        features = torch.cat((src_features, dst_features), dim=1)
        scores = torch.mm(features, self.W)
        log_probs = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(log_probs, y)

        return loss

    def evaluate(self, Z, test_edges, test_y):
        """
        Predict the test edges

        :param Z: final node embeddings
        :param test_edges: test edges
        :param test_y: test signs
        :return: auc, f1 scores, and loss
        """
        src_features = Z[test_edges[:, 0], :]
        dst_features = Z[test_edges[:, 1], :]
        features = torch.cat((src_features, dst_features), dim=1)
        scores = torch.mm(features, self.W)
        log_probs = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(log_probs, test_y)

        probs = torch.nn.functional.softmax(scores, dim=1)
        probs = probs.cpu().detach().numpy()
        predictions = np.argmax(probs, axis=1)
        scores = probs[:, 1]
        y = test_y.cpu().detach().numpy()

        auc, f1_scores = compute_accuracies(y, scores, predictions)
        return auc, f1_scores, loss
