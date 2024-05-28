# -*- coding: utf-8 -*-
# @Time   : 2022/1/18
# @Author : Peijie Sun
# @Email  : sun.hfut@gmail.com

r"""
SGL
################################################

Reference:
    Junliang Yu et al. "Graph Augmentation-Free Contrastive Learning for Recommendation.".

Reference code:
    https://github.com/Coder-Yu/QRec/blob/master/model/ranking/GACL.py
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

from tqdm import tqdm
import random 
import math
from collections import defaultdict

ssl_mode_dict = {
    0: "user_side",
    1: "item_side",
    2: "both_side",
    3: "merge"
}

augmentation_engine_dict = {
    0: 'gacl',
    1: 'sgl'
}

class GACL(GeneralRecommender):
    r"""Graph Augmentation-Free Contrastive Learning for Recommendation.

    It replace the complex data augmentation module in SGL with a simple but effective one. 

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(GACL, self).__init__(config, dataset)

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

        self.augmentation_engine = augmentation_engine_dict[config['augmentation_engine']]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # clusters embeddings
        self.dr_flag = config['dr_flag']
        self.mlp_flag = config['mlp_flag']

        if self.mlp_flag:
            self.dr_layer = nn.Sequential(
                nn.Linear(self.latent_dim, 2 * self.latent_dim, bias=False),
                nn.Tanh(),
                nn.Linear(2 * self.latent_dim, 2 * self.latent_dim, bias=False),
                nn.Tanh(),
                nn.Linear(2 * self.latent_dim, self.latent_dim, bias=False),
                nn.Tanh(),
                nn.Linear(self.latent_dim, config['dr_dim'], bias=False),
                nn.Tanh()
            )
        else:
            self.dr_layer = torch.nn.Linear(self.latent_dim, config['dr_dim'], bias=False)
        
        self.iteration = 0

        self.min_clusters = config['min_clusters']
        self.n_clusters = config['n_clusters']
        self.reg_cluster_weight = config['reg_cluster_weight']
        self.soft_cluster_weight = config['soft_cluster_weight']
        self.reg_cluster_num_weight = config['reg_cluster_num_weight']
        self.soft_flag = config['soft_flag']
        self.k_means_init_flag = config['k_means_init_flag']
        self.cluster_embedding = torch.randn(self.latent_dim, self.n_clusters).to(self.device)
        self.mse_loss = nn.MSELoss()

        self.trigger_flag = 0

        #import pdb; pdb.set_trace()
        #with torch.no_grad():
        #    self.cluster_embedding = self.KMeans(torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0))

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']

        self.device = config['device']

        self.user_np, self.item_np = self.dataset.inter_feat.numpy()[config['USER_ID_FIELD']], self.dataset.inter_feat.numpy()[config['ITEM_ID_FIELD']]
        self.n_nodes = self.n_users + self.n_items

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        '''
        #print('Looking for similar users and items for all users and items!')
        #self.user_similar_neighbors_mat, self.item_similar_neighbors_mat = self.get_similar_users_items(dataset, config)
        #import pdb; pdb.set_trace()

        # following variables are used to speedup sampling
        self.sub_graph_pool = config['sub_graph_pool']
        self.sub_mat_dict = self.prepare_sub_graphs()
        self.exist_sample_set = set()
        
        self.first_neighbors_dict = self.prepare_first_neighbors_dict()

        self.start_flag = True
        self.min_thre = config['min_thre']
        self.max_thre = config['max_thre']
        self.alpha = config['alpha']
        

        self.min_cluster_nodes = self.min_thre * (self.n_users + self.n_items) / (self.n_clusters) 
        self.max_cluster_nodes = self.max_thre * (self.n_users + self.n_items) / (self.n_clusters) 
        #'''

        self.ssl_strategy = config['ssl_strategy']
        self.positive_flag = config['positive_flag']

        self.neg_num = config['train_neg_sample_args']['by']
        self.eps = config['eps']

    def get_similar_users_items(self, dataset, config):
        # load parameters info
        self.k = config['k']
        self.shrink = config['shrink'] if 'shrink' in config else 0.0

        interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        shape = interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]

        user_similar_neighbors_mat = ComputeSimilarity(self, interaction_matrix, topk=self.k,
                                      shrink=self.shrink).compute_similarity('user')
        item_similar_neighbors_mat = ComputeSimilarity(self, interaction_matrix, topk=self.k,
                                      shrink=self.shrink).compute_similarity('item')

        return user_similar_neighbors_mat, item_similar_neighbors_mat

    def prepare_sub_graphs(self):
        aug_type = self.aug_type

        sub_mat_dict = {}
        for idx in tqdm(range(self.sub_graph_pool)):
            sub_mat_dict[idx] = self.get_norm_adj_mat(True, aug_type).to(self.device) if self.gpu_available else self.get_norm_adj_mat(True, aug_type)
        return sub_mat_dict

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

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def normalize(self, x_mat, dim=0):
        tmp_1 = torch.sqrt(torch.multiply(x_mat, x_mat).sum(dim=dim) + 1e-12)
        if dim == 0:
            x_mat = x_mat.transpose(0, 1)
            tmp_2 = x_mat / (tmp_1.reshape(-1, 1))
            tmp_2 = tmp_2.transpose(0, 1)
        elif dim == 1:
            tmp_2 = x_mat / (tmp_1.reshape(-1, 1))
        #import pdb; pdb.set_trace()
        return tmp_2

    def sample_subgraph_idx(self):
        idx_1 = np.random.randint(0, self.sub_graph_pool)
        idx_2 = np.random.randint(0, self.sub_graph_pool)

        while (idx_1, idx_2) in self.exist_sample_set:
            idx_1 = np.random.randint(0, self.sub_graph_pool)
            idx_2 = np.random.randint(0, self.sub_graph_pool)

        self.exist_sample_set.add((idx_1, idx_2))
        return self.sub_mat_dict[idx_1], self.sub_mat_dict[idx_2]

    def forward(self):
        aug_type = self.aug_type

        #'''
        # start to augment the input data
        if aug_type in [0, 1]:
            sub_mat = {}
            sub_mat_1, sub_mat_2 = self.sample_subgraph_idx()
            for k in range(self.n_layers):
                sub_mat['sub_mat_1_layer_%d' % k] = sub_mat_1
                sub_mat['sub_mat_2_layer_%d' % k] = sub_mat_2
        else:
            sub_mat = {}
            for k in range(self.n_layers):
                sub_mat['sub_mat_1_layer_%d' % k], sub_mat['sub_mat_2_layer_%d' % k] = self.sample_subgraph_idx()
        #'''
        
        all_embeddings = self.get_ego_embeddings()
        all_embeddings_sub1 = all_embeddings
        all_embeddings_sub2 = all_embeddings


        embeddings_list = [all_embeddings]
        embeddings_list_sub1 = [all_embeddings_sub1]
        embeddings_list_sub2 = [all_embeddings_sub2]

        #'''
        embeddings_list = [] 
        embeddings_list_sub1 = []
        embeddings_list_sub2 = []
        #'''

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

            #'''
            all_embeddings_sub1 = torch.sparse.mm(sub_mat['sub_mat_1_layer_%d' % layer_idx], all_embeddings_sub1)
            embeddings_list_sub1.append(all_embeddings_sub1)
            #''#'

            all_embeddings_sub2 = torch.sparse.mm(sub_mat['sub_mat_2_layer_%d' % layer_idx], all_embeddings_sub2)
            #all_embeddings_sub2 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings_sub2)
            embeddings_list_sub2.append(all_embeddings_sub2)
            #'''

        '''
        layers = random.randint(1, self.n_layers)
        for layer_idx in range(layers):
            all_embeddings_sub1 = torch.sparse.mm(sub_mat['sub_mat_1_layer_%d' % layer_idx], all_embeddings_sub1)
            embeddings_list_sub1.append(all_embeddings_sub1)
        #'''

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        self.user_all_embeddings, self.item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        #'''
        lightgcn_all_embeddings_sub1 = torch.stack(embeddings_list_sub1, dim=1)
        lightgcn_all_embeddings_sub1 = torch.mean(lightgcn_all_embeddings_sub1, dim=1)
        self.user_all_embeddings_sub1, self.item_all_embeddings_sub1 = torch.split(lightgcn_all_embeddings_sub1, [self.n_users, self.n_items])

        lightgcn_all_embeddings_sub2 = torch.stack(embeddings_list_sub2, dim=1)
        lightgcn_all_embeddings_sub2 = torch.mean(lightgcn_all_embeddings_sub2, dim=1)
        self.user_all_embeddings_sub2, self.item_all_embeddings_sub2 = torch.split(lightgcn_all_embeddings_sub2, [self.n_users, self.n_items])
        #'''

        if torch.isnan(torch.sum(self.user_embedding.weight)):
            import pdb; pdb.set_trace()

        return self.user_all_embeddings, self.item_all_embeddings

    def LightGCN_encoder(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [] 

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        return lightgcn_all_embeddings

    def perturbed_LightGCN_encoder(self, all_embeddings):
        embeddings_list = []
        for k in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            random_noise = torch.rand(all_embeddings.shape)
            all_embeddings += torch.multiply(torch.sign(all_embeddings).to(self.device), torch.nn.functional.normalize(random_noise, p=2, dim=1).to(self.device)) * self.eps
            embeddings_list.append(all_embeddings)

        perturbed_embeddings = torch.stack(embeddings_list, dim=1)
        perturbed_embeddings = torch.mean(perturbed_embeddings, dim=1)
        return torch.split(perturbed_embeddings, [self.n_users, self.n_items])

    def forward_gacl(self):
        lightgcn_all_embeddings = self.LightGCN_encoder()

        self.user_all_embeddings_sub1, self.item_all_embeddings_sub1 = self.perturbed_LightGCN_encoder(lightgcn_all_embeddings)
        self.user_all_embeddings_sub2, self.item_all_embeddings_sub2 = self.perturbed_LightGCN_encoder(lightgcn_all_embeddings)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, epoch, batch_idx, item_clusters=None):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        #import pdb; pdb.set_trace()

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        if self.augmentation_engine == 'gacl':
            user_all_embeddings, item_all_embeddings = self.forward_gacl()
        elif self.augmentation_engine == 'sgl':
            user_all_embeddings, item_all_embeddings = self.forward()

        #with torch.no_grad():
        #    count, _ = self.update_cluster(interaction, user_all_embeddings, item_all_embeddings)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]

        u_embeddings = u_embeddings.repeat(self.neg_num, 1)
        pos_embeddings = pos_embeddings.repeat(self.neg_num, 1)

        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        #'''
        # calculate SSL Loss
        if self.ssl_mode in ['user_side', 'item_side', 'both_side']:
            ssl_loss = self.calculate_ssl_loss_1(interaction)
        elif self.ssl_mode in ['merge']:
            ssl_loss = self.calculate_ssl_loss_2(interaction)
        else:
            raise ValueError("Invalid ssl_mode!")
        #'''

        '''
        if self.soft_flag:
            cluster_loss_func = self.calculate_soft_cluster_loss
        else:
            cluster_loss_func = self.calculate_hard_cluster_loss

        cluster_loss, item_clusters, _, ret_cluster_loss, ret_cluster_loss_2 = cluster_loss_func(user_all_embeddings, item_all_embeddings, interaction, epoch, batch_idx, item_clusters)

        #loss = mf_loss + self.reg_weight * reg_loss #+ self.reg_cluster_weight * cluster_loss  #+ ssl_loss
        #loss = cluster_loss
        #'''

        loss = mf_loss + self.reg_weight * reg_loss + ssl_loss
        
        #loss = self.reg_weight + ssl_loss

        '''
        if epoch == 1:
            import pdb; pdb.set_trace()

        if torch.isnan(torch.sum(self.user_embedding.weight)):
            import pdb; pdb.set_trace()
        #'''

        #import pdb; pdb.set_trace()

        #ssl_loss = torch.Tensor([0]).to(self.device)

        return loss, mf_loss, ssl_loss #, mf_loss, ret_cluster_loss, ret_cluster_loss_2, item_clusters
        
    # calculate the ssl loss with the ssl_mode ['user_id', 'item_side', 'both_side']
    def calculate_ssl_loss_1(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]

        user = torch.unique(user)
        pos_item = torch.unique(pos_item)

        if self.ssl_mode in ['user_side', 'both_side']:
            user_emb1 = self.user_all_embeddings_sub1[user]
            user_emb2 = self.user_all_embeddings_sub2[user]

            normalize_user_emb1 = torch.nn.functional.normalize(user_emb1, p=2, dim=1)
            normalize_user_emb2 = torch.nn.functional.normalize(user_emb2, p=2, dim=1)
            
            normalize_all_user_emb2 = torch.nn.functional.normalize(self.user_all_embeddings_sub2, p=2, dim=1)
            
            pos_score_user = torch.sum(torch.multiply(normalize_user_emb1, normalize_user_emb2), dim=1)
            #ttl_score_user = torch.matmul(normalize_user_emb1, normalize_user_emb2.transpose(0, 1))
            ttl_score_user = torch.matmul(normalize_user_emb1, normalize_all_user_emb2.transpose(0, 1))

            pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
            ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)

            ssl_loss_user = -torch.mean(torch.log(pos_score_user / ttl_score_user))
        
        if self.ssl_mode in ['item_side', 'both_side']:
            item_emb1 = self.item_all_embeddings_sub1[pos_item]
            item_emb2 = self.item_all_embeddings_sub2[pos_item]

            normalize_item_emb1 = torch.nn.functional.normalize(item_emb1, p=2, dim=1)
            normalize_item_emb2 = torch.nn.functional.normalize(item_emb2, p=2, dim=1)
            
            normalize_all_item_emb2 = torch.nn.functional.normalize(self.item_all_embeddings_sub2, p=2, dim=1)
            
            pos_score_item = torch.sum(torch.multiply(normalize_item_emb1, normalize_item_emb2), dim=1)
            #ttl_score_item = torch.matmul(normalize_item_emb1, normalize_item_emb2.transpose(0, 1))
            ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.transpose(0, 1))
            
            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)

            ssl_loss_item = -torch.mean(torch.log(pos_score_item / ttl_score_item))

        if self.ssl_mode == 'user_side':
            ssl_loss = self.ssl_reg * ssl_loss_user
        elif self.ssl_mode == 'item_side':
            ssl_loss = self.ssl_reg * ssl_loss_item
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)

        return ssl_loss

    def neighbor_sample(self, input_list):
        if len(input_list) == 1:
            return input_list[0]
        else:
            random.shuffle(input_list)
            return input_list[0]

    def construct_batch_dict(self, interaction):
        batch_user_dict, batch_item_dict = defaultdict(list), defaultdict(list)
        batch_user_list = interaction[self.USER_ID].cpu().numpy().tolist()
        batch_item_list = interaction[self.ITEM_ID].cpu().numpy().tolist()

        for idx, user in enumerate(batch_user_list):
            item = batch_item_list[idx]
            batch_user_dict[user].append(item)
            batch_item_dict[item].append(user)

        return batch_user_dict, batch_item_dict

    # calculate the ssl loss with the ssl_mode ['merge']
    def calculate_ssl_loss_2(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        
        batch_user_dict, batch_item_dict = self.construct_batch_dict(interaction)

        #import pdb; pdb.set_trace() 
  
        # Original Algorithm: following is original ssl loss calculation function
        if self.ssl_strategy == 0:
            # following code is used for original ssl loss computation
            #batch_users = torch.unique(user)
            batch_users = user
            user_emb1 = self.user_all_embeddings_sub1[batch_users]
            user_emb2 = self.user_all_embeddings_sub2[batch_users]

            #batch_items = torch.unique(pos_item)
            batch_items = pos_item
            item_emb1 = self.item_all_embeddings_sub1[batch_items]
            item_emb2 = self.item_all_embeddings_sub2[batch_items]

            emb_merge1 = torch.cat([user_emb1, item_emb1], dim=0)
            emb_merge2 = torch.cat([user_emb2, item_emb2], dim=0)

            # cosine similarity
            normalize_emb_merge1 = torch.nn.functional.normalize(emb_merge1, p=2, dim=1)
            normalize_emb_merge2 = torch.nn.functional.normalize(emb_merge2, p=2, dim=1)

            pos_score = torch.sum(torch.multiply(normalize_emb_merge1, normalize_emb_merge2), dim=1)
            ttl_score = torch.matmul(normalize_emb_merge1, normalize_emb_merge2.transpose(0, 1))

            '''
            with torch.no_grad():
                #self.similarity_matrix[]
                _, topk_idx = torch.topk(ttl_score, 500, dim=-1)  # n_users x k
                sample_matrix = torch.zeros_like(ttl_score, dtype=torch.int)
                row_idx = torch.range(0, topk_idx.shape[0]-1).reshape(-1, 1).tile(500).reshape(-1, 1).long()
                column_idx = topk_idx.reshape(-1, 1)
                sample_matrix[row_idx, column_idx] = 1.
            #import pdb; pdb.set_trace()
            ttl_score = torch.multiply(ttl_score, sample_matrix)
            #'''

            #'''
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss
            #'''

        # First Possible Positive Algorithm - For Sparse Data, like amazon instant video: 
        # following is enhanced ssl loss calculation function
        # multiple possible positives, please refer to Supervised Contrastive Learning, NeurIPS 2021. 
        # following is the possible strategy, which is time-consuming
        elif self.ssl_strategy == 1:
            batch_users_o3 = []
            batch_items_o3 = []
            batch_users_e3 = []
            batch_items_e3 = []

            batch_users_4 = []
            batch_items_4 = []

            with torch.no_grad():
                batch_users_list = user.cpu().numpy().tolist()
                batch_items_list = pos_item.cpu().numpy().tolist()
                for idx, user in enumerate(batch_users_list):
                    batch_users_o3.append(user)
                    batch_users_e3.append(batch_items_list[idx])

                    batch_users_4.append(user)
                    batch_users_4.append(user)

                for idx, item in enumerate(batch_items_list):
                    batch_items_o3.append(item)
                    batch_items_e3.append(batch_users_list[idx])

                    batch_items_4.append(item)
                    batch_items_4.append(item)

                batch_users_o3 = torch.tensor(batch_users_o3).long().to(self.device)
                batch_items_o3 = torch.tensor(batch_items_o3).long().to(self.device)
                batch_users_e3 = torch.tensor(batch_users_e3).long().to(self.device)
                batch_items_e3 = torch.tensor(batch_items_e3).long().to(self.device)

                batch_users_4 = torch.tensor(batch_users_4).long().to(self.device)
                batch_items_4 = torch.tensor(batch_items_4).long().to(self.device)

            #import pdb; pdb.set_trace()

            user_emb_o3 = self.user_all_embeddings_sub1[batch_users_o3]
            item_emb_o3 = self.item_all_embeddings_sub1[batch_items_o3]

            user_emb_e3 = self.item_all_embeddings_sub1[batch_users_e3]
            item_emb_e3 = self.user_all_embeddings_sub1[batch_items_e3]

            user_emb3 = torch.cat([user_emb_o3, user_emb_e3], dim=1).reshape(-1, user_emb_o3.shape[1])
            item_emb3 = torch.cat([item_emb_o3, item_emb_e3], dim=1).reshape(-1, user_emb_o3.shape[1])

            #import pdb; pdb.set_trace()

            user_emb4 = self.user_all_embeddings_sub2[batch_users_4]
            item_emb4 = self.item_all_embeddings_sub2[batch_items_4]

            emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
            emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

            # cosine similarity
            normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
            normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)

            pos_score = torch.sum(torch.multiply(normalize_emb_merge3, normalize_emb_merge4), dim=1)
            #pos_score = pos_score.reshape(-1, self.k_values + 1)
            ttl_score = torch.matmul(normalize_emb_merge3, normalize_emb_merge4.transpose(0, 1))

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss
            #'''

            #import pdb; pdb.set_trace()

        elif self.ssl_strategy == 2:
            batch_users = torch.unique(user)
            batch_items = torch.unique(pos_item)

            batch_users_3 = []
            batch_items_3 = []

            batch_users_4 = []
            batch_items_4 = []

            with torch.no_grad():
                batch_users_list = batch_users.cpu().numpy().tolist()
                batch_items_list = (batch_items + self.n_users).cpu().numpy().tolist()
                for idx, user in enumerate(batch_users_list):
                    batch_users_3.append(user)
                    batch_users_4.append(user)

                    for item in batch_user_dict[user]:
                        batch_users_3.append(item)
                        batch_users_4.append(user)

                for idx, item in enumerate(batch_items_list):
                    batch_items_3.append(item)
                    batch_items_4.append(item)

                    for user in batch_item_dict[item]:
                        batch_items_3.append(user)
                        batch_items_4.append(item)

                batch_users_3 = torch.tensor(batch_users_3).long().to(self.device)
                batch_items_3 = torch.tensor(batch_items_3).long().to(self.device)

                batch_users_4 = torch.tensor(batch_users_4).long().to(self.device)
                batch_items_4 = torch.tensor(batch_items_4).long().to(self.device)

                #self.similarity_matrix[]
                '''
                _, topk_idx = torch.topk(ttl_score, 500, dim=-1)  # n_users x k
                sample_matrix = torch.zeros_like(ttl_score, dtype=torch.int)
                row_idx = torch.range(0, topk_idx.shape[0]-1).reshape(-1, 1).tile(500).reshape(-1, 1).long()
                column_idx = topk_idx.reshape(-1, 1)
                sample_matrix[row_idx, column_idx] = 1.
                #'''

            node_all_embeddings_sub1 = torch.cat([self.user_all_embeddings_sub1, self.item_all_embeddings_sub1], dim=0)
            node_all_embeddings_sub2 = torch.cat([self.user_all_embeddings_sub2, self.item_all_embeddings_sub2], dim=0)

            user_emb3 = node_all_embeddings_sub1[batch_users_3]
            item_emb3 = node_all_embeddings_sub1[batch_items_3]

            user_emb4 = node_all_embeddings_sub2[batch_users_4]
            item_emb4 = node_all_embeddings_sub2[batch_items_4]

            emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
            emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

            # cosine similarity
            normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
            normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)

            pos_score = torch.sum(torch.multiply(normalize_emb_merge3, normalize_emb_merge4), dim=1)
            pos_score = torch.exp(pos_score / self.ssl_temp)#.reshape(-1, 2)

            ttl_score = torch.matmul(normalize_emb_merge3, normalize_emb_merge4.transpose(0, 1))
            #ttl_score = torch.multiply(ttl_score, sample_matrix)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            #import pdb; pdb.set_trace()

            # pos_score: (num_batch_nodes, self.KNN + 1)
            '''
            if self.positive_flag == 0:
                pos_score = torch.sum(pos_score, dim=1)
                ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            elif self.positive_flag == 1:
                ssl_loss = -torch.mean(torch.sum(torch.log(pos_score / ttl_score), dim=1))
            #'''
            
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss

        # Second Remove False Negative Algorithm:
        ### following script is used to remove false negative from the denominator
        elif self.ssl_strategy == 3:
            with torch.no_grad():
                batch_node_list = torch.cat([batch_users, batch_items + self.n_users], dim=0).cpu().numpy().tolist()

                batch_node_clusters = []
                for node in batch_node_list:
                    batch_node_clusters.append(self.cluster_dict[node])

                batch_node_clusters = torch.Tensor(batch_node_clusters).long().to(self.device)            
                
                denominator_mask = batch_node_clusters.reshape(-1, 1) - batch_node_clusters.reshape(1, -1)
                denominator_mask = (denominator_mask != 0).float()
            #import pdb; pdb.set_trace()

            ttl_score = torch.multiply(ttl_score, denominator_mask)
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss

            #import pdb; pdb.set_trace()

        elif self.ssl_strategy == 4:
            #batch_users = torch.unique(user)
            batch_users = user
            user_emb1 = self.user_all_embeddings_sub1[batch_users]
            user_emb2 = self.user_all_embeddings_sub2[batch_users]

            #batch_items = torch.unique(pos_item)
            batch_items = pos_item
            item_emb1 = self.item_all_embeddings_sub1[batch_items]
            item_emb2 = self.item_all_embeddings_sub2[batch_items]

            emb_merge1 = torch.cat([user_emb1, item_emb1], dim=0)
            emb_merge2 = torch.cat([user_emb2, item_emb2], dim=0)

            # cosine similarity
            normalize_emb_merge1 = torch.nn.functional.normalize(emb_merge1, p=2, dim=1)
            normalize_emb_merge2 = torch.nn.functional.normalize(emb_merge2, p=2, dim=1)

            pos_score = torch.ones().to(self.device)
            ttl_score = torch.matmul(normalize_emb_merge1, normalize_emb_merge2.transpose(0, 1))

            #'''
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss

        elif self.ssl_strategy == 5:
            batch_users_o3 = []
            batch_items_o3 = []
            batch_users_e3 = []
            batch_items_e3 = []

            batch_users_4 = []
            batch_items_4 = []

            with torch.no_grad():
                batch_users_list = user.cpu().numpy().tolist()
                batch_items_list = pos_item.cpu().numpy().tolist()
                for idx, user in enumerate(batch_users_list):
                    batch_users_o3.append(user)
                    batch_users_e3.append(batch_items_list[idx])
                    batch_users_o3.append(self.neighbor_sample(self.user_similar_neighbors_mat[user]))

                    batch_users_4.append(user)
                    batch_users_4.append(user)
                    batch_users_4.append(user)

                for idx, item in enumerate(batch_items_list):
                    batch_items_o3.append(item)
                    batch_items_e3.append(batch_users_list[idx])
                    batch_items_o3.append(self.neighbor_sample(self.item_similar_neighbors_mat[item]))

                    batch_items_4.append(item)
                    batch_items_4.append(item)
                    batch_items_4.append(item)

                batch_users_o3 = torch.tensor(batch_users_o3).long().to(self.device)
                batch_items_o3 = torch.tensor(batch_items_o3).long().to(self.device)
                batch_users_e3 = torch.tensor(batch_users_e3).long().to(self.device)
                batch_items_e3 = torch.tensor(batch_items_e3).long().to(self.device)

                batch_users_4 = torch.tensor(batch_users_4).long().to(self.device)
                batch_items_4 = torch.tensor(batch_items_4).long().to(self.device)

            #import pdb; pdb.set_trace()

            user_emb_o3 = self.user_all_embeddings_sub1[batch_users_o3].reshape(interaction[self.USER_ID].shape[0], -1)
            item_emb_o3 = self.item_all_embeddings_sub1[batch_items_o3].reshape(interaction[self.USER_ID].shape[0], -1)

            user_emb_e3 = self.item_all_embeddings_sub1[batch_users_e3].reshape(interaction[self.USER_ID].shape[0], -1)
            item_emb_e3 = self.user_all_embeddings_sub1[batch_items_e3].reshape(interaction[self.USER_ID].shape[0], -1)

            user_emb3 = torch.cat([user_emb_o3, user_emb_e3], dim=1).reshape(-1, self.user_all_embeddings_sub1.shape[1])
            item_emb3 = torch.cat([item_emb_o3, item_emb_e3], dim=1).reshape(-1, self.user_all_embeddings_sub1.shape[1])

            #import pdb; pdb.set_trace()

            user_emb4 = self.user_all_embeddings_sub2[batch_users_4]
            item_emb4 = self.item_all_embeddings_sub2[batch_items_4]

            emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
            emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

            # cosine similarity
            normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
            normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)

            pos_score = torch.sum(torch.multiply(normalize_emb_merge3, normalize_emb_merge4), dim=1)
            #pos_score = pos_score.reshape(-1, self.k_values + 1)
            ttl_score = torch.matmul(normalize_emb_merge3, normalize_emb_merge4.transpose(0, 1))

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        if self.augmentation_engine == 'gacl':
            user_all_embeddings, item_all_embeddings = self.forward_gacl()
        elif self.augmentation_engine == 'sgl':
            user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            if self.augmentation_engine == 'gacl':
                self.restore_user_e, self.restore_item_e = self.forward_gacl()
            elif self.augmentation_engine == 'sgl':
                self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)