# -*- coding: utf-8 -*-
# @Time   : 2021/12/09
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
User_based
################################################

Reference:
    Badrul Sarwar et al. "Analysis of Recommendation Algorithms for ECommerce." in Electronic Commerce 2000.

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


class User_based(GeneralRecommender):
    r"""

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(User_based, self).__init__(config, dataset)

        # load dataset info
        self.dataset = dataset
        #self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.config = config

        # generate intermediate data
        self.item_scores = self.get_norm_adj_mat_without_rating().to(self.device)

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']

        self.item_cnt = torch.zeros(self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False)
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def get_norm_adj_mat_with_rating(self, is_subgraph=False, aug_type=None):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        user_np, item_np = self.dataset.inter_feat.numpy()['user_id'], self.dataset.inter_feat.numpy()['item_id']
        ratings = self.dataset.inter_feat.numpy()['rating']
        interactions = np.ones_like(user_np, dtype=np.float32)

        interaction_matrix = sp.csr_matrix((interactions, (user_np, item_np)), shape=(self.n_users, self.n_items))
        rating_matrix = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.n_users, self.n_items))

        avg_rating = np.array(rating_matrix.mean(1)) # (n_users, 1)
        rating_matrix = rating_matrix - interaction_matrix.multiply(avg_rating) # (n_users, n_items); r_ai - mean(r_a)

        tmp_x_1 = rating_matrix * rating_matrix.transpose() # (n_users, n_users) 

        tmp_x_2 = sp.linalg.norm(rating_matrix, axis=1).reshape(-1, 1) # n_users, 1; sqrt(\sum_{i\R_a}(r_ai - mean(r_a))^2)
        tmp_x_2 = np.matmul(tmp_x_2, np.transpose(tmp_x_2)) # n_users, n_users; sqrt(\sum_{i\R_a}(r_ai - mean(r_a))^2) sqrt(\sum_{i\R_b}(r_bi - mean(r_b))^2)

        sim_uu = tmp_x_1 / tmp_x_2

        rank_uu = np.argsort(-sim_uu, axis=1) # Descending along the rows, the most similar users rank first

        # filter the top-K nearest neighbors for each user
        topK = 100
        topk_index = [x.item() for x in rank_uu[:, topK]]
        topk_value = np.array([sim_uu[idx, index] for idx, index in enumerate(topk_index)]).reshape(-1, 1)
        k_nearest_neighbors = ((sim_uu - topk_value) >= 0) # k-nearest neighbors, with boolen values

        tmp_x3 = np.multiply(sim_uu, k_nearest_neighbors) # n_users, n_users
        item_scores = tmp_x3 * interaction_matrix # n_users, n_items

        return torch.FloatTensor(item_scores)

    def get_norm_adj_mat_without_rating(self, is_subgraph=False, aug_type=None):
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

        tmp_x_1 = interaction_matrix * interaction_matrix.transpose() # (n_users, n_users) 

        tmp_x_2 = sp.linalg.norm(interaction_matrix, axis=1).reshape(-1, 1) # n_users, 1; sqrt(\sum_{i\R_a}(r_ai - mean(r_a))^2)
        tmp_x_2 = np.matmul(tmp_x_2, np.transpose(tmp_x_2)) # n_users, n_users; sqrt(\sum_{i\R_a}(r_ai - mean(r_a))^2) sqrt(\sum_{i\R_b}(r_bi - mean(r_b))^2)

        sim_uu = tmp_x_1 / tmp_x_2

        rank_uu = np.argsort(-sim_uu, axis=1) # Descending along the rows, the most similar users rank first

        # filter the top-K nearest neighbors for each user
        topK = 100
        topk_index = [x.item() for x in rank_uu[:, topK]]
        topk_value = np.array([sim_uu[idx, index] for idx, index in enumerate(topk_index)]).reshape(-1, 1)
        k_nearest_neighbors = ((sim_uu - topk_value) >= 0) # k-nearest neighbors, with boolen values

        tmp_x3 = np.multiply(sim_uu, k_nearest_neighbors) # n_users, n_users
        item_scores = tmp_x3 * interaction_matrix # n_users, n_items

        return torch.FloatTensor(item_scores)

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
        #import pdb; pdb.set_trace()
        scores = self.item_scores[user]

        import pdb; pdb.set_trace()
        return scores.view(-1)