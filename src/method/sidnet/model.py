#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn import Parameter
from method.sidnet.decoder import Decoder


class SidNet(torch.nn.Module):
    def __init__(self,
                 hid_dims,
                 in_dim,
                 device,
                 num_nodes,
                 num_layers=2,
                 num_diff_layers=10,
                 c=0.15):
        """
        Constructor of SidNet

        :param hid_dims: hidden dimension
        :param in_dim: input dimension
        :param device: gpu device
        :param num_nodes: number of nodes (n)
        :param num_layers: number of layers (L)
        :param num_diff_layers: number of diffusion steps (K)
        :param c: ratio of local feature injection
        """
        super(SidNet, self).__init__()

        self.hid_dims = hid_dims
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_diff_layers = num_diff_layers
        self.device = device
        self.c = c
        self.num_nodes = num_nodes
        self.act = torch.tanh

        self.setup_layers()

    def setup_layers(self):
        """
        Set up layers
        :return: none
        """
        self.sidnet_layer = SidNetLayer(device=self.device,
                                        num_diff_layers=self.num_diff_layers,
                                        c=self.c)

        self.decoder = Decoder(emb_size=self.hid_dims[-1])

        self.Ws = torch.nn.ParameterList()
        self.Wx = torch.nn.ParameterList()
        self.bns = torch.nn.ModuleList()

        self.Ws.append(Parameter(torch.FloatTensor(self.in_dim, self.hid_dims[0])))
        self.Wx.append(Parameter(torch.FloatTensor(self.hid_dims[0] * 2, self.hid_dims[0])))
        self.bns.append(torch.nn.BatchNorm1d(num_features=self.hid_dims[0]))

        for i in range(1, self.num_layers):
            self.Ws.append(Parameter(torch.FloatTensor(self.hid_dims[i-1],
                                                       self.hid_dims[i])))
            self.Wx.append(Parameter(torch.FloatTensor(self.hid_dims[i-1] * 2,
                                                       self.hid_dims[i])))
            self.bns.append(torch.nn.BatchNorm1d(num_features=self.hid_dims[i]))

        for ws, wx in zip(self.Ws, self.Wx):
            torch.nn.init.xavier_normal_(ws)
            torch.nn.init.xavier_normal_(wx)

    def forward(self, nApT, nAmT, X, edges, y):
        """
        Forward X into loss

        :param nApT: transposed normalized matrix for + sign
        :param nAmT: transposed normalized matrix for - sign
        :param X: input node features
        :param edges: edges
        :param y: signs
        :return: BCE loss
        """

        prev_X = X
        for i in range(0, self.num_layers):
            pred_X = torch.mm(X, self.Ws[i])

            P, M = self.sidnet_layer(nApT, nAmT, pred_X)
            X = torch.cat((P, M), dim=1)
            X = torch.mm(X, self.Wx[i])

            # skip connection
            if i > 0:
                X = X + prev_X

            X = self.bns[i](X)
            X = self.act(X)

            prev_X = X

        self.Z = X
        loss = self.decoder(self.Z, edges, y)

        return loss

    def evaluate(self, test_edges, test_y):
        """
        Evaluate test edges in terms of AUC and F1-macro

        :param test_edges: test edges
        :param test_y: test signs

        :return: scores
        """
        return self.decoder.evaluate(self.Z, test_edges, test_y)


class SidNetLayer(torch.nn.Module):
    def __init__(self,
                 device,
                 num_diff_layers=10,
                 c=0.15):
        """
        Constructore of Sid layer

        :param device: gpu device
        :param num_diff_layers: number of diffusion steps (K)
        :param c: restart probability
        """
        super(SidNetLayer, self).__init__()

        self.device = device
        self.num_diff_layers = num_diff_layers
        self.c = c

    def forward(self, nApT, nAmT, X):
        """
        Forward X into P and M

        :param nApT: transposed normalized matrix for + sign
        :param nAmT: transposed normalized matrix for - sign
        :param X: local node features

        :return: P (positive embeddings) and M (negative embeddings)
        """
        old_P = X
        old_M = torch.FloatTensor(X.shape).uniform_(-1.0, 1.0).to(self.device)

        new_P = old_P
        new_M = old_M

        tilda_X = self.c * X

        for i in range(self.num_diff_layers):
            new_P = torch.sparse.mm(nApT, old_P) + torch.sparse.mm(nAmT, old_M) + tilda_X
            new_M = torch.sparse.mm(nAmT, old_P) + torch.sparse.mm(nApT, old_M)

            old_P = new_P
            old_M = new_M

        return new_P, new_M
