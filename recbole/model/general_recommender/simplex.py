# -*- coding: utf-8 -*-
# @Time   : 2022/01/20
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
SimpleX
################################################

Reference:
    Kelong Mao et al. "SimpleX: A Simple and Strong Baseline for Collaborative Filtering." in CIKM 2021.

Reference code:
    https://github.com/xue-pai/TwoTowers
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class SimpleX(GeneralRecommender):
    r"""
        Introduction of SimpleX 
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimpleX, self).__init__(config, dataset)

        # load dataset info
        self.dataset = dataset
        #self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of UltraGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.w1 = config['w1']
        self.w2 = config['w2']
        self.w3 = config['w3']
        self.w4 = config['w4']

        self.negative_weight = config['negative_weight']
        self.gamma = config['gamma']
        self.lambda_ = config['lambda']
        self.initial_weight = config['initial_weight']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.initial_weights()

        train_mat = self.get_ui_constraint_mat()
        self.get_ii_constraint_mat(train_mat, config['ii_neighbor_num'])

    def initial_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embedding.weight, std=self.initial_weight)
    
    def get_ii_constraint_mat(self, train_mat, num_neighbors, ii_diagonal_zero=False):
        G = train_mat.T.dot(train_mat) # G = A^T * A
        n_items = G.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            G[range(n_items), range(n_items)] = 0
        items_D = np.sum(G, axis=0).reshape(-1)

        omega_iD = (np.sqrt(items_D + 1) / (items_D + 1)).reshape(-1, 1) # sqrt(g_i)/(g_i - G_{i,i})
        omega_jD = (1 / np.sqrt(items_D + 1)).reshape(1, -1) # 1/sqrt(g_j)
        all_ii_constraint_mat = torch.from_numpy(omega_iD.dot(omega_jD)) # (1/(g_i - G_{i,i}))*(sqrt(g_i)/sqrt(g_j))
        for idx in range(n_items):
            row = all_ii_constraint_mat[idx] * torch.from_numpy(G.getrow(idx).toarray()[0]) # (G_{i,j}/(g_i - G_{i,i}))*(sqrt(g_i)/sqrt(g_j))
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[idx] = row_idxs
            res_sim_mat[idx] = row_sims
            if idx % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(idx))
        print('Computation \\Omega OK!')
        
        self.ii_constraint_mat = res_sim_mat.float()
        self.ii_neighbor_mat = res_mat.long()

    def get_ui_constraint_mat(self):
        user_np, item_np = self.dataset.inter_feat.numpy()['user_id'], self.dataset.inter_feat.numpy()['item_id']

        train_mat = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        for idx, _ in enumerate(user_np):
            train_mat[user_np[idx], item_np[idx]] = 1.0

        items_D = np.sum(train_mat, axis=0).reshape(-1)
        users_D = np.sum(train_mat, axis=1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
        constraint_mat = constraint_mat.flatten()

        self.constraint_mat = constraint_mat
        return train_mat

    def get_betas(self, users, pos_items, neg_items):
        if self.w2 > 0:
            pos_weight = self.constraint_mat[users * self.n_items + pos_items].to(self.device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(self.device)
        
        users = (users * self.n_items).unsqueeze(0)
        if self.w4 > 0:
            import pdb; pdb.set_trace()
            neg_weight = self.constraint_mat[torch.cat([users] * neg_items.size(0)).transpose(1, 0) + neg_items].flatten().to(self.device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(0)).to(self.device)

        beta_weights = torch.cat((pow_weight, neg_weight))
        return beta_weights

    def cal_loss_L(self, users, pos_items, neg_items, beta_weights):
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(self.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(self.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        neighbor_embeds = self.item_embedding(self.ii_neighbor_mat[pos_items].to(self.device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(self.device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embedding(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        import pdb; pdb.set_trace()

        betas_weight = self.get_betas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, betas_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        u_embeddings = self.user_embedding(user)
        i_embeddings = self.item_embedding(item)
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        # get user embedding from storage variable
        u_embeddings = self.user_embedding[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.item_embedding.weight.transpose(0, 1))

        return scores.view(-1)
