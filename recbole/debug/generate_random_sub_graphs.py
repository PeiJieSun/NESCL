# -*- coding: utf-8 -*-
# @Time   : 2021/10/24
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
SGL
################################################

Reference:
    Jiancan Wu et al. "Self-supervised Graph Learning for Recommendation." in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

# following toolkit comes from https://github.com/wujcan/SGL
from recbole.util.cython.random_choice import randint_choice

from tqdm import tqdm

import random 
import math
from collections import defaultdict


class Generate_Sub_Graph(GeneralRecommender):
    r"""SGL: Self-supervised Graph learning for Recommendation.

    The idea of SGL is to supplement the classical supervised task of recommendation 
    with anauxiliary self-supervised task, which reinforces node representation learning
    via self-discrimination. 

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(Generate_Sub_Graph, self).__init__(config, dataset)

        # load dataset info
        self.dataset = dataset
        #self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_np, self.item_np = self.dataset.inter_feat.numpy()[config['USER_ID_FIELD']], self.dataset.inter_feat.numpy()[config['ITEM_ID_FIELD']]
        self.n_nodes = self.n_users + self.n_items

        self.ssl_ratio = config['ssl_ratio']

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']

    def get_norm_adj_mat(self, is_subgraph=False, aug_type=0):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """

        if is_subgraph and aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = randint_choice(self.n_users, size=self.n_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.n_items, size=self.n_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(self.user_np, dtype=np.float32), (self.user_np, self.item_np)), 
                    shape=(self.n_users, self.n_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.n_users)), shape=(self.n_nodes, self.n_nodes))
            elif aug_type in [1, 2]:
                '''
                np.random.shuffle(self.nums)
                keep_idx = np.array(range(len(self.user_np)))[self.nums.astype(int)==1].tolist()
                '''
                keep_idx = randint_choice(len(self.user_np), size=int(len(self.user_np) * (1 - self.ssl_ratio)), replace=False)

                user_np = np.array(self.user_np)[keep_idx]
                item_np = np.array(self.item_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(self.n_nodes, self.n_nodes))
                #'''
            else:
                raise ValueError("Invalid aug_type!")
        else:
            user_np = np.array(self.user_np)
            item_np = np.array(self.item_np)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(self.n_nodes, self.n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print('use the pre adjcency matrix')

        adj_matrix = adj_matrix.tocoo()

        index = torch.LongTensor([adj_matrix.row, adj_matrix.col])
        data = torch.FloatTensor(adj_matrix.data)
        SparseL = torch.sparse.FloatTensor(index, data, torch.Size(adj_matrix.shape))

        return SparseL

    def forward(self):

        aug_type = 1
        sub_mat_dict = {}
        for idx in tqdm(range(500)):
            sub_mat_dict[idx] = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)
        return sub_mat_dict