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
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from sklearn.preprocessing import normalize

#import pdb; pdb.set_trace()

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

class ComputeSimilarity:

    def __init__(self, model, dataMatrix, topk=10, shrink=0, normalize=True):
        r"""Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.model = model

    def compute_similarity(self, method, block_size=100):
        r"""Compute the similarity for the given dataset

        Args:
            method (str) : Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            block_size (int): divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\_columns \div block\_size`. 

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num], 
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        user_similar_neighbors_mat, item_similar_neighbors_mat = [], []
        user_similar_neighbors_weights_mat, item_similar_neighbors_weights_mat = [], []

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if method == 'user':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        elif method == 'item':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:

            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if method == 'user':
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray().squeeze()

            if data.ndim == 1:
                data = np.expand_dims(data, axis=1)
            
            # Compute similarities

            if method == 'user':
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_line_weights = this_block_weights.squeeze()
                else:
                    this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = sumOfSquared[Index] * sumOfSquared + self.shrink + 1e-6
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users or items
                # - Partition the data to extract the set of relevant users or items
                # - Sort only the relevant users or items
                # - Get the original index
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[0:self.TopK]

                relevant_partition_sorting = np.argsort(-this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                tmp_values = this_line_weights[top_k_idx][notZerosMask]
                
                '''
                if self.model.renormalize_similarity:
                    if len(tmp_values) != 0:
                        tmp_values = np.array(tmp_values).reshape(1, -1)
                        tmp_values = normalize(tmp_values, norm='l1', axis=1)
                        tmp_values = tmp_values[0].tolist()
                        #import pdb; pdb.set_trace()
                #'''

                if method == 'user':
                    user_similar_neighbors_mat.append(top_k_idx[notZerosMask])
                    user_similar_neighbors_weights_mat.append(tmp_values)
                else:
                    item_similar_neighbors_mat.append(top_k_idx[notZerosMask])
                    item_similar_neighbors_weights_mat.append(tmp_values)

            start_block += block_size

        if method == 'user':
            return user_similar_neighbors_mat, user_similar_neighbors_weights_mat
        elif method == 'item':
            return item_similar_neighbors_mat, item_similar_neighbors_weights_mat

class SUPCCL(GeneralRecommender):
    r"""SGL: Self-supervised Graph learning for Recommendation.

    The idea of SGL is to supplement the classical supervised task of recommendation 
    with anauxiliary self-supervised task, which reinforces node representation learning
    via self-discrimination. 

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SUPCCL, self).__init__(config, dataset)

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
        
        self.iteration = 0

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
        
        self.ssl_strategy = config['ssl_strategy']
        self.positive_cl_type = config['positive_cl_type']

        self.neg_num = config['train_neg_sample_args']['by']

        self.lightgcn_flag = config['lightgcn_flag']
        self.pairwise_loss = config['pairwise_loss']
        self.random_strategy = config['random_strategy']
        self.augmentation = config['augmentation']
        self.add_initial_embedding = config['add_initial_embedding']
        #self.set_batch_nodes = config['set_batch_nodes']
        #self.add_more_neighbors = config['add_more_neighbors']
        self.interacted_neighbors = config['interacted_neighbors']
        self.similar_user_neighbors = config['similar_user_neighbors']
        self.similar_item_neighbors = config['similar_item_neighbors']
        self.different_view = config['different_view']

        self.different_view_weight = config['different_view_weight']
        self.interacted_neighbors_weight = config['interacted_neighbors_weight']
        self.sample_item_weight = config['sample_item_weight']
        self.sample_user_weight = config['sample_user_weight']

        self.sample_item_weight_flag = config['sample_item_weight_flag']
        self.sample_user_weight_flag = config['sample_user_weight_flag']

        self.supcon_flag = config['supcon_flag']

        #self.renormalize_similarity = config['renormalize_similarity']
        self.prob_sampling = config['prob_sampling']

        if self.lightgcn_flag == True:
            assert self.ssl_strategy == 0
            assert self.augmentation == False

        #import pdb; pdb.set_trace()

        if self.ssl_strategy != 0:
            print('Looking for similar users and items for all users and items!')
            self.user_similar_neighbors_mat, self.user_similar_neighbors_weights_mat, \
            self.item_similar_neighbors_mat, self.item_similar_neighbors_weights_mat = self.get_similar_users_items(dataset, config)
        
        # following variables are used to speedup sampling
        if self.augmentation:
            self.sub_graph_pool = config['sub_graph_pool']
            self.sub_mat_dict = self.prepare_sub_graphs()
            self.exist_sample_set = set()

    def get_similar_users_items(self, dataset, config):
        # load parameters info
        self.k = config['k']
        self.shrink = config['shrink'] if 'shrink' in config else 0.0

        interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        shape = interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]

        user_similar_neighbors_mat, user_similar_neighbors_weights_mat = ComputeSimilarity(self, interaction_matrix, topk=self.k,
                                      shrink=self.shrink).compute_similarity('user')
        item_similar_neighbors_mat, item_similar_neighbors_weights_mat = ComputeSimilarity(self, interaction_matrix, topk=self.k,
                                      shrink=self.shrink).compute_similarity('item')

        return user_similar_neighbors_mat, user_similar_neighbors_weights_mat, item_similar_neighbors_mat, item_similar_neighbors_weights_mat
    
    def prepare_first_neighbors_dict(self):
        first_neighbors_dict = defaultdict(set)
        
        for idx, user in enumerate(self.user_np):
            item = self.item_np[idx] + self.n_users
            first_neighbors_dict[user].add(item)
            first_neighbors_dict[item].add(user)
        return first_neighbors_dict

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
        if self.augmentation:
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

        if self.add_initial_embedding:
            embeddings_list = [all_embeddings]
            embeddings_list_sub1 = [all_embeddings_sub1]
            embeddings_list_sub2 = [all_embeddings_sub2]
        elif not self.add_initial_embedding:
            embeddings_list = [] 
            embeddings_list_sub1 = []
            embeddings_list_sub2 = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

            #'''
            if self.augmentation:
                all_embeddings_sub1 = torch.sparse.mm(sub_mat['sub_mat_1_layer_%d' % layer_idx], all_embeddings_sub1)
                embeddings_list_sub1.append(all_embeddings_sub1)

                all_embeddings_sub2 = torch.sparse.mm(sub_mat['sub_mat_2_layer_%d' % layer_idx], all_embeddings_sub2)
                embeddings_list_sub2.append(all_embeddings_sub2)
            #'''

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        self.user_all_embeddings, self.item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        #'''
        if self.augmentation:
            lightgcn_all_embeddings_sub1 = torch.stack(embeddings_list_sub1, dim=1)
            lightgcn_all_embeddings_sub1 = torch.mean(lightgcn_all_embeddings_sub1, dim=1)
            self.user_all_embeddings_sub1, self.item_all_embeddings_sub1 = torch.split(lightgcn_all_embeddings_sub1, [self.n_users, self.n_items])

            lightgcn_all_embeddings_sub2 = torch.stack(embeddings_list_sub2, dim=1)
            lightgcn_all_embeddings_sub2 = torch.mean(lightgcn_all_embeddings_sub2, dim=1)
            self.user_all_embeddings_sub2, self.item_all_embeddings_sub2 = torch.split(lightgcn_all_embeddings_sub2, [self.n_users, self.n_items])
        #'''

        #import pdb; pdb.set_trace()

        if torch.isnan(torch.sum(self.user_embedding.weight)):
            import pdb; pdb.set_trace()

        return self.user_all_embeddings, self.item_all_embeddings

    def calculate_loss(self, interaction, epoch, batch_idx, item_clusters=None):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        #import pdb; pdb.set_trace()

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

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

        if self.lightgcn_flag == False:
            #'''
            # calculate SSL Loss
            if self.ssl_mode in ['user_side', 'item_side', 'both_side']:
                ssl_loss = self.calculate_ssl_loss_1(interaction)
            elif self.ssl_mode in ['merge']:
                ssl_loss = self.calculate_ssl_loss_2(interaction)
            else:
                raise ValueError("Invalid ssl_mode!")
            #'''

            #import pdb; pdb.set_trace()
            if self.pairwise_loss:
                loss = mf_loss + self.reg_weight * reg_loss + ssl_loss
            else:
                loss = self.reg_weight + ssl_loss
        else:
            loss = mf_loss + self.reg_weight * reg_loss
            ssl_loss = torch.Tensor([0]).to(self.device)
        
        #loss = self.reg_weight + ssl_loss

        #import pdb; pdb.set_trace()

        #ssl_loss = torch.Tensor([0]).to(self.device)

        return loss, mf_loss, ssl_loss


    def neighbor_sample(self, input_list, weight_list):
        if len(input_list) == 1:
            return input_list[0], weight_list[0]
        else:
            if self.prob_sampling:
                prob = np.asarray(weight_list).astype('float64')
                prob = prob / sum(prob)
                #import pdb; pdb.set_trace()
                idx = np.random.choice(range(0, len(input_list)), size=1, replace=True, p=prob)
                idx = idx.item()
                #import pdb; pdb.set_trace()
            else:
                idx = np.random.randint(0, len(input_list))
            #random.shuffle(input_list)
            return input_list[idx], weight_list[idx]

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

        # interaction is the input data
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        
        #batch_user_dict, batch_item_dict = self.construct_batch_dict(interaction)
  
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

            #'''
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss
            #'''


        # SupCL two kinds of loss functions 
        elif self.ssl_strategy == 9:
            if self.augmentation:
                node_all_embeddings_sub1 = torch.cat([self.user_all_embeddings_sub1, self.item_all_embeddings_sub1], dim=0)
                node_all_embeddings_sub2 = torch.cat([self.user_all_embeddings_sub2, self.item_all_embeddings_sub2], dim=0)
            else:
                node_all_embeddings_sub1 = torch.cat([self.user_all_embeddings, self.item_all_embeddings], dim=0)
                node_all_embeddings_sub2 = torch.cat([self.user_all_embeddings, self.item_all_embeddings], dim=0)

            batch_user_weight = []
            batch_item_weight = []

            # batch_users_3 is used to index the user embedding from view-2
            batch_users_3 = []
            # batch_items_3 is used to index the user embedding from view-2
            batch_items_3 = []

            # batch_users_4 is used to index the user embedding from view-1
            batch_users_4 = []
            # batch_items_4 is used to index the user embedding from view-1
            batch_items_4 = []

            batch_nodes_list = []

            with torch.no_grad():
                batch_users_list = user.cpu().numpy().tolist()
                # update item ids to map the original item id to the constructed graph
                batch_items_list = (pos_item+self.n_users).cpu().numpy().tolist()

                # batch_nodes_list stores both the batch_users_list and the batch_item_list
                batch_nodes_list.extend(batch_users_list)
                batch_nodes_list.extend(batch_items_list)

                for idx, user in enumerate(batch_users_list):
                    if self.different_view:
                        batch_user_weight.append(self.different_view_weight)

                        # user itself is also treated as the positive sample, \mathbf{h}_i''
                        batch_users_3.append(user)
                        # \mathbf{h}_i'
                        batch_users_4.append(user)

                    # add user-item positive pair
                    if self.interacted_neighbors:
                        item = batch_items_list[idx]
                        batch_user_weight.append(self.interacted_neighbors_weight)
                        # interacted item is treated as the positive sample, \mathbf{h}_k''
                        batch_users_3.append(item)
                        # \mathbf{h}_i'
                        batch_users_4.append(user)

                    # add user and her k-nearest neighbors positive pair
                    if self.similar_user_neighbors:
                        if self.random_strategy:
                            # the prob_sampling used in self.neighbor_sample controls whether sampling the collaborative neighbors with their probabilities 
                            sample_user, sample_weight = self.neighbor_sample(self.user_similar_neighbors_mat[user], self.user_similar_neighbors_weights_mat[user])

                            # sample_user_weight_flag controls whether add weights to the collaborative neighbors 
                            if self.sample_user_weight_flag:
                                batch_user_weight.append(sample_weight * self.sample_user_weight)
                            else:
                                batch_user_weight.append(self.sample_user_weight)

                            batch_users_3.append(sample_user)
                            batch_users_4.append(user)

                            #if self.add_more_neighbors:
                            #    sample_item = self.neighbor_sample(self.item_similar_neighbors_mat[item-self.n_users])+self.n_users

                            #    batch_users_3.append(item)
                            #    batch_users_4.append(user)
                        else:
                            for idx, sample_user in enumerate(self.user_similar_neighbors_mat[user]):
                                batch_users_3.append(sample_user)
                                batch_users_4.append(user)

                                if self.sample_user_weight_flag:
                                    sample_weight = self.user_similar_neighbors_weights_mat[user][idx]
                                    batch_user_weight.append(sample_weight * self.sample_user_weight)

                for idx, item in enumerate(batch_items_list):
                    if self.different_view:
                        
                        batch_item_weight.append(self.different_view_weight)

                        batch_items_3.append(item)
                        batch_items_4.append(item)

                    # add item-user positive pair
                    if self.interacted_neighbors:
                        user = batch_users_list[idx]

                        batch_item_weight.append(self.interacted_neighbors_weight)

                        batch_items_3.append(user)
                        batch_items_4.append(item)

                    # add item and its k-nearest neighbors positive pair
                    if self.similar_item_neighbors:
                        if self.random_strategy:
                            sample_item, sample_weight = self.neighbor_sample(self.item_similar_neighbors_mat[item-self.n_users], self.item_similar_neighbors_weights_mat[item-self.n_users])
                            sample_item += +self.n_users

                            if self.sample_item_weight_flag:
                                batch_item_weight.append(sample_weight * self.sample_item_weight)
                            else:
                                batch_item_weight.append(self.sample_item_weight)

                            batch_items_3.append(sample_item)
                            batch_items_4.append(item)

                            #if self.add_more_neighbors:
                            #    sample_user = self.neighbor_sample(self.user_similar_neighbors_mat[user])

                            #    batch_items_3.append(sample_user)
                            #    batch_items_4.append(user)
                        else:
                            for idx, sample_item in enumerate(self.item_similar_neighbors_mat[item-self.n_users]):
                                sample_item += self.n_users

                                batch_items_3.append(sample_item)
                                batch_items_4.append(item)

                                if self.sample_item_weight_flag:
                                    sample_weight = self.item_similar_neighbors_weights_mat[item-self.n_users][idx]
                                    batch_item_weight.append(sample_weight * self.sample_item_weight)

                batch_users_3 = torch.tensor(batch_users_3).long().to(self.device)
                batch_items_3 = torch.tensor(batch_items_3).long().to(self.device)

                batch_users_4 = torch.tensor(batch_users_4).long().to(self.device)
                batch_items_4 = torch.tensor(batch_items_4).long().to(self.device)

                #if self.set_batch_nodes:
                #    batch_nodes_list = torch.tensor(list(set(batch_nodes_list))).long().to(self.device)
                #else:
                batch_nodes_list = torch.tensor(list(batch_nodes_list)).long().to(self.device)


            if self.supcon_flag:
                # batch_users_3, batch_items_3 are consisf of different positive samples, get representations from view-2
                user_emb3 = node_all_embeddings_sub2[batch_users_3]
                item_emb3 = node_all_embeddings_sub2[batch_items_3]

                # batch_users_4, batch_items_4 are consisf of the anchor nodes themseleves, get representations from view-1
                user_emb4 = node_all_embeddings_sub1[batch_users_4]
                item_emb4 = node_all_embeddings_sub1[batch_items_4]

                # get representations from view-2
                batch_node_emb = node_all_embeddings_sub2[batch_nodes_list]

                emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
                emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

                # cosine similarity
                normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
                normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)
                normalize_batch_node_emb = torch.nn.functional.normalize(batch_node_emb, p=2, dim=1)

                # the anchor nodes' representations from view-1 mutliply differeent kinds of positive samples from view-2 
                pos_score = torch.sum(torch.multiply(normalize_emb_merge4, normalize_emb_merge3), dim=1)

                # the anchor nodes' representations from view-1 matmul the negative samples from view-2
                ttl_score = torch.matmul(normalize_emb_merge4, normalize_batch_node_emb.transpose(0, 1))
            
            else:
                # batch_users_3, batch_items_3 are consisf of different positive samples, get representations from view-1
                user_emb3 = node_all_embeddings_sub1[batch_users_3]
                item_emb3 = node_all_embeddings_sub1[batch_items_3]

                # batch_users_4, batch_items_4 are consisf of the anchor nodes themseleves, get representations from view-2
                user_emb4 = node_all_embeddings_sub2[batch_users_4]
                item_emb4 = node_all_embeddings_sub2[batch_items_4]

                # get representations from view-2
                batch_node_emb = node_all_embeddings_sub2[batch_nodes_list]
                
                emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
                emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

                # cosine similarity
                normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
                normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)
                normalize_batch_node_emb = torch.nn.functional.normalize(batch_node_emb, p=2, dim=1)

                # differeent kinds of positive samples from view-1 mutliply the anchor nodes' representations from view-2
                pos_score = torch.sum(torch.multiply(normalize_emb_merge3, normalize_emb_merge4), dim=1)
                #pos_score = pos_score.reshape(-1, self.k_values + 1)

                # different kinds of positive samples from view-1 matmul the negative samples from view-2
                ttl_score = torch.matmul(normalize_emb_merge3, normalize_batch_node_emb.transpose(0, 1))
            
            #import pdb; pdb.set_trace()


            #batch_node_weight = torch.Tensor(batch_user_weight+batch_item_weight).float().cuda()

            #pos_score = pos_score * batch_node_weight

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            #import pdb; pdb.set_trace()

            # it corresponds to the supcl-in version, log every element
            if self.positive_cl_type == 1:
                ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            # it corresponds to the supcl-out version, log the sum of the same node
            elif self.positive_cl_type == 2:
                if self.random_strategy:
                    pos_score = pos_score.reshape(-1, 3)
                    ttl_score = ttl_score.reshape(-1, 3)
                else:
                    pos_score = pos_score.reshape(-1, 2 + self.k)
                    ttl_score = ttl_score.reshape(-1, 2 + self.k)
                ssl_loss = -torch.mean(torch.log(torch.mean(pos_score / ttl_score, dim=1)))
            
            ssl_loss = self.ssl_reg * ssl_loss

        return ssl_loss

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