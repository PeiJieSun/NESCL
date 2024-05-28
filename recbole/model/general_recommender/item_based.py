# -*- coding: utf-8 -*-
# @Time   : 2021/12/09
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
Co-clustering
################################################

Reference:
    M. Deshpande et al. "Item-based top-N recommendation algorithms." in TOIS 2004.

"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F 
import random

from collections import defaultdict

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class Item_based(GeneralRecommender):
    r"""

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(Item_based, self).__init__(config, dataset)

        # load dataset info
        self.dataset = dataset
        #self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.config = config

        # generate intermediate data
        self.item_scores = self.get_norm_adj_mat().to(self.device)

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']

        self.item_cnt = torch.zeros(self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def get_norm_adj_mat(self, is_subgraph=False, aug_type=None):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        user_np, item_np = self.dataset.inter_feat.numpy()['user_id'], self.dataset.inter_feat.numpy()['item_id']
        interactions = np.ones_like(user_np, dtype=np.float32)

        interaction_matrix = sp.csr_matrix((interactions, (user_np, item_np)), shape=(self.n_users, self.n_items))

        tmp_x_1 = interaction_matrix.transpose() * interaction_matrix # co-occurrence of two items

        tmp_x_2 = sp.linalg.norm(interaction_matrix, axis=0).reshape(-1, 1) # n_items, 1; 
        tmp_x_4 = np.matmul(tmp_x_2, np.transpose(tmp_x_2)) # n_items, n_items; 

        sim_ii = tmp_x_1 / tmp_x_4 - np.identity(self.n_items) # n_items, n_items

        #import pdb; pdb.set_trace()
        
        rank_ii = np.argsort(-sim_ii, axis=1) # Descending along the rows, the most similar items rank first

        # filter the top-K nearest neighbors for each item
        topK = 100
        topk_index = [x.item() for x in rank_ii[:, topK]]

        topk_value = np.array([sim_ii[idx, index] for idx, index in enumerate(topk_index)]).reshape(-1, 1)

        k_nearest_neighbors = ((sim_ii - topk_value) >= 0) # k-nearest neighbors, with boolen values

        print([idx for idx in range(1683) if k_nearest_neighbors[1, idx]==True])
        

        scores = interaction_matrix * np.transpose(np.multiply(k_nearest_neighbors, sim_ii))
        scores[np.isnan(scores)] = 0
        #scores = interaction_matrix * np.transpose(k_nearest_neighbors)

        #import pdb; pdb.set_trace()

        return torch.FloatTensor(scores)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        # dot with all item embedding to accelerate
        #import pdb; pdb.set_trace()
        scores = self.item_scores[user]

        return scores.view(-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        # dot with all item embedding to accelerate
        scores = self.item_scores[user]

        return scores