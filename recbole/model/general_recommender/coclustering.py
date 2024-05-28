# -*- coding: utf-8 -*-
# @Time   : 2021/10/24
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
Co-clustering
################################################

Reference:
    Mathilde Caron et al. "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments." in NeurIPS 2020.

Reference code:
    https://github.com/facebookresearch/swav
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F 
import random

from collections import defaultdict

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

drop_flag_dict = {
    0: True,
    1: False
}

class CoClustering(GeneralRecommender):
    r"""
    Co-clustering - first - Experimental MODE

    Our first attempt to utilize the co-clustering technique to regularize the parameter optimizing procedure. 

    https://www.notion.so/min-cut-cc9123c1218d48408775492bc5841f31

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CoClustering, self).__init__(config, dataset)

        # load dataset info
        self.dataset = dataset
        #self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.co_reg = config['co_reg'] # float32 type: 
        self.temp_1 = config['temp_1']
        self.temp_2 = config['temp_2']
        self.beta_1 = config['beta_1']
        self.beta_2 = config['beta_2']
        self.drop_flag = drop_flag_dict[config['drop_flag']]
        self.nmb_prototypes = config['nmb_prototypes']
        self.proto_gap = config['proto_gap']
        self.min_rating_ratio = config['min_rating_ratio']
        self.ssl_ratio = config['ssl_ratio']

        self.drop = nn.Dropout(p=1-self.proto_gap)
        self.config = config
        self.sigmoid = nn.Sigmoid()

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # prototype layer
        self.prototypes = nn.Linear(self.latent_dim, self.nmb_prototypes, bias=False)

        # count the minimize ratings of each user
        user_rating_dict = defaultdict(list)
        user_list = self.dataset['user_id'].tolist()
        rating_list = self.dataset['rating'].tolist()
        for idx, user in enumerate(user_list):
            user_rating_dict[user].append(rating_list[idx])
        for user in user_rating_dict:
            user_rating_dict[user] = (min(user_rating_dict[user]) / 0.25 + 1) / 5 / self.min_rating_ratio
        self.user_rating_dict = user_rating_dict

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']

    def get_norm_adj_mat(self, is_subgraph=False, aug_type=None):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        self.user_np, self.item_np = self.dataset.inter_feat.numpy()['user_id'], self.dataset.inter_feat.numpy()['item_id']
        ratings = np.ones_like(self.user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_items

        if aug_type == 1:
            tmp_idx_list = list(range(len(self.user_np)))
            random.shuffle(tmp_idx_list)
            keep_idx = tmp_idx_list[:int(len(self.user_np) * (1 - self.ssl_ratio))]
            #import pdb; pdb.set_trace()
            #keep_idx = randint_choice(len(self.user_np), size=int(len(self.user_np) * (1 - self.ssl_ratio)), replace=False)

            user_np = np.array(self.user_np)[keep_idx]
            item_np = np.array(self.item_np)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(self.user_np)
            item_np = np.array(self.item_np)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        # D^(-0.5) * A * D^(-0.5)
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

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        # get the embedding of users and items with the original graph
        all_embeddings = self.get_ego_embeddings()

        embeddings_list = [all_embeddings] 

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, flag=False):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        #import pdb; pdb.set_trace()

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, self.prototypes.weight)

        co_clustering_loss = self.calculate_co_clustering_loss(interaction, flag)

        #import pdb; pdb.set_trace()
        #loss = mf_loss + self.reg_weight * reg_loss + self.co_reg * co_clustering_loss
        loss = co_clustering_loss

        #import pdb; pdb.set_trace()

        return loss, mf_loss, co_clustering_loss
    
    def calculate_co_clustering_loss(self, interaction, flag=False):
        # Execute the clustering strategy
        # If the projection_head is not none, execture the projection_head_layer 

        # get the embedding of users and item with the augmented graph
        sub_mat = {}
        for k in range(self.n_layers):
            sub_mat['sub_mat_1_layer_%d' % k] = self.get_norm_adj_mat(True, 1).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, 1)
        
        all_embeddings_sub1 = self.get_ego_embeddings()
        embeddings_list_sub1 = [all_embeddings_sub1]

        for layer_idx in range(self.n_layers):
            all_embeddings_sub1 = torch.sparse.mm(sub_mat['sub_mat_1_layer_%d' % layer_idx], all_embeddings_sub1)
            embeddings_list_sub1.append(all_embeddings_sub1)

        lightgcn_all_embeddings_sub1 = torch.stack(embeddings_list_sub1, dim=1)
        lightgcn_all_embeddings_sub1 = torch.mean(lightgcn_all_embeddings_sub1, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings_sub1, [self.n_users, self.n_items])

        '''
        if self.projection_head is not None:
            user_all_embeddings = self.projection_head(user_all_embeddings)
            item_all_embeddings = self.projection_head(item_all_embeddings)
        #'''

        # Execute the clustering layer
        user_all_embeddings = self.prototypes(user_all_embeddings)
        item_all_embeddings = self.prototypes(item_all_embeddings)

        user_all_embeddings = F.relu(user_all_embeddings)
        item_all_embeddings = F.relu(item_all_embeddings)

        # Execute the normalize layer
        user_all_embeddings = user_all_embeddings / (torch.sum(user_all_embeddings, dim=1, keepdims=True) + 1e-12)
        item_all_embeddings = item_all_embeddings / (torch.sum(item_all_embeddings, dim=1, keepdims=True) + 1e-12)

        if self.drop_flag:
            user_all_embeddings = F.relu(self.drop(user_all_embeddings))
            item_all_embeddings = F.relu(self.drop(item_all_embeddings))
        else:
            user_all_embeddings_mask = user_all_embeddings > 1/(self.nmb_prototypes * self.proto_gap)
            item_all_embeddings_mask = item_all_embeddings > 1/(self.nmb_prototypes * self.proto_gap)

            user_all_embeddings = F.relu(user_all_embeddings * user_all_embeddings_mask)
            item_all_embeddings = F.relu(item_all_embeddings * item_all_embeddings_mask)

        user_all_embeddings = user_all_embeddings / (torch.sum(user_all_embeddings, dim=1, keepdims=True) + 1e-12)
        item_all_embeddings = item_all_embeddings / (torch.sum(item_all_embeddings, dim=1, keepdims=True) + 1e-12)

        min_clusters = torch.min(torch.sum(torch.cat([user_all_embeddings, item_all_embeddings], dim=0), dim=0))
        max_clusters = torch.max(torch.sum(torch.cat([user_all_embeddings, item_all_embeddings], dim=0), dim=0))

        # calculate Co-clustering Loss
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        
        m_up = (interaction['rating'] / 0.25 + 1) / 5

        m_un = []
        for u in user.cpu().tolist():
            m_un.append(self.user_rating_dict[u])
        m_un = torch.FloatTensor(np.array(m_un)).to(self.device)

        #loss_1 = torch.sum(torch.abs(u_embeddings - pos_embeddings)) / min_clusters
        #loss_1 = (- torch.log(torch.sigmoid(-torch.square(torch.sum(u_embeddings * pos_embeddings, dim=1) - m_up) / self.temp_1)) + torch.log(torch.sigmoid((1 - m_up) / self.temp_1))
        #        ).mean() * max_clusters / min_clusters
        loss_1 = (- torch.log(torch.sigmoid(-torch.square(torch.sum(u_embeddings * pos_embeddings, dim=1)) / self.temp_1)) + torch.log(torch.sigmoid((1 - m_up) / self.temp_1))
                ).mean() * max_clusters / min_clusters
        #loss_2 = torch.sum(torch.abs(u_embeddings + neg_embeddings)) / min_clusters
        #loss_2 = (- torch.log(torch.sigmoid((m_un - torch.sum(u_embeddings * neg_embeddings, dim=1)) / self.temp_2)) + torch.log(torch.sigmoid(m_un / self.temp_2))
        #        ).mean() * max_clusters / min_clusters
        loss_2 = (- torch.log(torch.sigmoid((- torch.sum(u_embeddings * neg_embeddings, dim=1)) / self.temp_2)) + torch.log(torch.sigmoid(m_un / self.temp_2))
                ).mean() * max_clusters / min_clusters
        
        #loss = self.beta_1 * loss_1 + self.beta_2 * loss_2 
        loss = loss_1 + loss_2

        #import pdb; pdb.set_trace()

        #if flag:
        #    import pdb; pdb.set_trace()

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)