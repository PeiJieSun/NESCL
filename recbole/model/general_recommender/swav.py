# -*- coding: utf-8 -*-
# @Time   : 2021/10/24
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
SwAV
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

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

# following toolkit comes from https://github.com/wujcan/SGL
from recbole.util.cython.random_choice import randint_choice

ssl_mode_dict = {
    0: "user_side",
    1: "item_side",
    2: "both_side"
}

l2_norm_dict = {
    0: True,
    1: False
}

# 0: rating prediction with user/item representations after prototypes
# 1: rating prediction with user/item representations before prototypes
cluster_flag_dict = {
    0: True, 
    1: False
}

unique_flag_dict = {
    0: True,
    1: False
}

class SwAV(GeneralRecommender):
    r"""SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.

    SwAV is an efficient and simple method for pre-training convnets without using annotations. 
    Similarly to contrastive approaches, SwAV learns representations by comparing transformations 
    of an image, but unlike contrastive methods, it does not require to compute feature pairwise 
    comparisons. It makes our framework more efficient since it does not require a large memory 
    bank or an auxiliary momentum network. Specifically, our method simultaneously clusters the 
    data while enforcing consistency between cluster assignments produced for different augmentations 
    (or “views”) of the same image, instead of comparing features directly. Simply put, we use a 
    “swapped” prediction mechanism where we predict the cluster assignment of a view from the 
    representation of another view. Our method can be trained with large and small batches and can 
    scale to unlimited amounts of data.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SwAV, self).__init__(config, dataset)

        # load dataset info
        self.dataset = dataset
        #self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.ssl_ratio = config['ssl_ratio'] # float32 type: 
        self.ssl_temp = config['ssl_temp'] # float32 type: 
        self.ssl_reg = config['ssl_reg'] # float32 type: 
        # 0: 'user_side', 1: 'item_side', 2: 'both_side', 3: 'merge'
        self.ssl_mode = ssl_mode_dict[config['ssl_mode']] # str type: different kinds of self-supervised learning mode
        self.aug_type = config['aug_type'] # int type: different kinds of data augmentation

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

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']

        self.device = config['device']

        # normalize output features
        self.l2norm = l2_norm_dict[config['normalize']]

        self.cluster_prediction_flag = cluster_flag_dict[config['cluster_prediction_flag']]

        self.unique_flag = unique_flag_dict[config['unique_flag']]

        # projection head
        output_dim = config['feat_dim']
        hidden_mlp = config['hidden_mlp']
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(self.latent_dim, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        nmb_prototypes = config['nmb_prototypes']
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        self.user_embedding_queue = torch.zeros(config['num_aug'], config['queue_length'], config['feat_dim']).cuda()
        self.item_embedding_queue = torch.zeros(config['num_aug'], config['queue_length'], config['feat_dim']).cuda()
        self.merge_embedding_queue = torch.zeros(config['num_aug'], config['queue_length'], config['feat_dim']).cuda()

        self.config = config

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
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.n_users)), shape=(n_nodes, n_nodes))
            elif aug_type in [1, 2]:
                keep_idx = randint_choice(len(self.user_np), size=int(len(self.user_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(self.user_np)[keep_idx]
                item_np = np.array(self.item_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
            else:
                raise ValueError("Invalid aug_type!")
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
        aug_type = self.aug_type

        # start to augment the input data, each graph should be augmented two times, and the same nodes among the two augmented graphs are positive pairs. 
        # sub_mat_1, sub_mat_2 are corresponing to twice augmentation matrix. When aug_type is 0 or 1, the sub_mat_* among different gcn layers are the same.
        # And when aug_type is 2, the sub_mat_* among different gcn layers are different. 
        # For different epochs, we should re-augment the input user-item bipartite graph.
        if aug_type in [0, 1]:
            sub_mat = {}
            sub_mat_1 = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)
            sub_mat_2 = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)
            for k in range(self.n_layers):
                sub_mat['sub_mat_1_layer_%d' % k] = sub_mat_1
                sub_mat['sub_mat_2_layer_%d' % k] = sub_mat_2
        else:
            sub_mat = {}
            for k in range(self.n_layers):
                sub_mat['sub_mat_1_layer_%d' % k] = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)
                sub_mat['sub_mat_2_layer_%d' % k] = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)

        all_embeddings = self.get_ego_embeddings()
        all_embeddings_sub1 = all_embeddings
        all_embeddings_sub2 = all_embeddings

        embeddings_list = [all_embeddings]
        embeddings_list_sub1 = [all_embeddings_sub1]
        embeddings_list_sub2 = [all_embeddings_sub2]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

            all_embeddings_sub1 = torch.sparse.mm(sub_mat['sub_mat_1_layer_%d' % layer_idx], all_embeddings_sub1)
            embeddings_list_sub1.append(all_embeddings_sub1)

            all_embeddings_sub2 = torch.sparse.mm(sub_mat['sub_mat_2_layer_%d' % layer_idx], all_embeddings_sub2)
            embeddings_list_sub2.append(all_embeddings_sub2)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        lightgcn_all_embeddings_sub1 = torch.stack(embeddings_list_sub1, dim=1)
        lightgcn_all_embeddings_sub1 = torch.mean(lightgcn_all_embeddings_sub1, dim=1)
        self.user_all_embeddings_sub1, self.item_all_embeddings_sub1 = torch.split(lightgcn_all_embeddings_sub1, [self.n_users, self.n_items])

        lightgcn_all_embeddings_sub2 = torch.stack(embeddings_list_sub2, dim=1)
        lightgcn_all_embeddings_sub2 = torch.mean(lightgcn_all_embeddings_sub2, dim=1)
        self.user_all_embeddings_sub2, self.item_all_embeddings_sub2 = torch.split(lightgcn_all_embeddings_sub2, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        #'''
        if self.cluster_prediction_flag:
            if self.projection_head is not None:
                u_embeddings = self.projection_head(u_embeddings)
                pos_embeddings = self.projection_head(pos_embeddings)
                neg_embeddings = self.projection_head(neg_embeddings)
            if self.l2norm:
                u_embeddings = F.normalize(u_embeddings, dim=1, p=2)
                pos_embeddings = F.normalize(pos_embeddings, dim=1, p=2)
                neg_embeddings = F.normalize(neg_embeddings, dim=1, p=2)
            u_embeddings = self.prototypes(u_embeddings)
            pos_embeddings = self.prototypes(pos_embeddings)
            neg_embeddings = self.prototypes(neg_embeddings)
        #'''
        
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        # calculate SSL Loss
        #'''
        if self.ssl_mode in ['user_side', 'item_side', 'both_side']:
            ssl_loss = self.calculate_swav_loss(interaction)
        else:
            raise ValueError("Invalid ssl_mode!")
        #'''

        loss = mf_loss + self.reg_weight * reg_loss + self.ssl_reg * ssl_loss

        return loss
    
    # 1. First, normalize the learned representations, self.user_all_embeddings_sub1, self.item_all_embeddings_sub1, self.user_all_embeddings_sub2, self.item_all_embeddings_sub2
    # 2. Second, compute a code q_nt from this feature by mapping z_nt to a set of K trainable prototypes vectors, {c_1, ..., c_K}，the mapping function can be linear or multi-layer perceptrons. 

    # following code refers to link: https://github.com/facebookresearch/swav/blob/main/main_swav.py
    # calculate swav loss for 'user-side', 'item-side', 'both-side'
    def calculate_swav_loss(self, interaction):
        if self.unique_flag:
            user = torch.unique(interaction[self.USER_ID])
            pos_item = torch.unique(interaction[self.ITEM_ID])
        else:
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]

        if self.ssl_mode in ['user_side', 'both_side']:
            user_embeddings_sub1 = self.user_all_embeddings_sub1[user]
            user_embeddings_sub2 = self.user_all_embeddings_sub2[user]
        if self.ssl_mode in ['item_side', 'both_side']:
            pos_item_embeddings_sub1 = self.item_all_embeddings_sub1[pos_item]
            pos_item_embeddings_sub2 = self.item_all_embeddings_sub2[pos_item]

        if self.projection_head is not None:
            if self.ssl_mode in ['user_side', 'both_side']:
                user_embeddings_sub1 = self.projection_head(user_embeddings_sub1)
                user_embeddings_sub2 = self.projection_head(user_embeddings_sub2)
            if self.ssl_mode in ['item_side', 'both_side']:
                pos_item_embeddings_sub1 = self.projection_head(pos_item_embeddings_sub1)
                pos_item_embeddings_sub2 = self.projection_head(pos_item_embeddings_sub2)

        if self.l2norm:
            if self.ssl_mode in ['user_side', 'both_side']:
                user_embeddings_sub1 = F.normalize(user_embeddings_sub1, dim=1, p=2)
                user_embeddings_sub2 = F.normalize(user_embeddings_sub2, dim=1, p=2)
            if self.ssl_mode in ['item_side', 'both_side']:
                pos_item_embeddings_sub1 = F.normalize(pos_item_embeddings_sub1, dim=1, p=2)
                pos_item_embeddings_sub2 = F.normalize(pos_item_embeddings_sub2, dim=1, p=2)

        if self.ssl_mode in ['user_side', 'both_side']:
            user_prototypes_sub1 = self.prototypes(user_embeddings_sub1)
            user_prototypes_sub2 = self.prototypes(user_embeddings_sub2)

            user_all_embeddings = [user_embeddings_sub1, user_embeddings_sub2]
            user_output = [user_prototypes_sub1, user_prototypes_sub2]
        if self.ssl_mode in ['item_side', 'both_side']:
            pos_item_prototypes_sub1 = self.prototypes(pos_item_embeddings_sub1)
            pos_item_prototypes_sub2 = self.prototypes(pos_item_embeddings_sub2)

            item_all_embeddings = [pos_item_embeddings_sub1, pos_item_embeddings_sub2]
            item_output = [pos_item_prototypes_sub1, pos_item_prototypes_sub2]

        loss = 0
        user_bs = user.shape[0]
        item_bs = pos_item.shape[0]

        for i, aug_id in enumerate(range(self.config['num_aug'])):
            with torch.no_grad():
                # fill the queue
                if self.ssl_mode in ['user_side', 'both_side']:
                    user_out = user_output[i]

                    self.user_embedding_queue[i, user_bs:] = self.user_embedding_queue[i, :-user_bs].clone()
                    self.user_embedding_queue[i, :user_bs] = user_all_embeddings[i]

                    user_out = torch.mm(
                        self.user_embedding_queue[i],
                        self.prototypes.weight.t()
                    )

                    # get assignments
                    user_q = self.distributed_sinkhorn(user_out)[-user_bs:]

                if self.ssl_mode in ['item_side', 'both_side']:
                    item_out = item_output[i]

                    self.item_embedding_queue[i, item_bs:] = self.item_embedding_queue[i, :-item_bs].clone()
                    self.item_embedding_queue[i, :item_bs] = item_all_embeddings[i]

                    item_out = torch.mm(
                        self.item_embedding_queue[i],
                        self.prototypes.weight.t()
                    )

                    # get assignments
                    item_q = self.distributed_sinkhorn(item_out)[-item_bs:]

            # cluster assignment prediction
            if self.ssl_mode in ['user_side', 'both_side']:
                user_subloss = 0
                for v in np.delete(np.arange(self.config['num_aug']), aug_id):
                    x = user_output[i] / self.config['temperature']
                    user_subloss -= torch.mean(torch.sum(user_q * F.log_softmax(x, dim=1), dim=1))
                loss += user_subloss / (self.config['num_aug'] - 1)

            if self.ssl_mode in ['item_side', 'both_side']:
                item_subloss = 0
                for v in np.delete(np.arange(np.sum(self.config['num_aug'])), aug_id):
                    x = item_output[i] / self.config['temperature']
                    item_subloss -= torch.mean(torch.sum(item_q * F.log_softmax(x, dim=1), dim=1))
                loss += item_subloss / (self.config['num_aug'] - 1)

        loss /= self.config['num_aug']
        return loss
    
    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.config['epsilon']).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.config['sinkhorn_iterations']):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

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

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out