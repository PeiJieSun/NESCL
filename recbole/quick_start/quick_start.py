# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger

import os
import torch
import pickle

from collections import defaultdict
import numpy as np 

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


def run_trial(model=None, dataset=None, config_file_list=None, config_dict=None, saved=False, RECEIVED_PARAMS=None, train_setp=None, eval_step=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config, nni_mode=True)
    logger = getLogger()

    logger.info(config)
    
    config.update(RECEIVED_PARAMS)
    config.update({'show_progress': False})

    train_data, valid_data, test_data = load_split_dataloaders(config['dataloader_file'])

    if train_setp:
        train_data.step = train_setp
    if eval_step:
        valid_data.step = eval_step
        test_data.step = eval_step

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress'], nni_mode=True
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, dataloader_file=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    #dataloader_file = None

    # first split the dataset
    if not dataloader_file:
        # dataset filtering
        dataset = create_dataset(config)
        if config['save_dataset']:
            dataset.save()
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)
        import pdb; pdb.set_trace()
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # load data from the saved-dataset and saved-dataloader
    else:
        train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

    if train_data.neg_sample_num !=  config['train_neg_sample_args']['by']:
        train_data.neg_sample_num = config['train_neg_sample_args']['by']
    
    #import pdb; pdb.set_trace()

    assert 'train_data_step' in config
    #import pdb; pdb.set_trace()
    if train_data.step != config['train_data_step']:
        train_data.step = config['train_data_step']

    if valid_data.step != config['val_data_step']:
        valid_data.step = config['val_data_step']
    
    if test_data.step != config['test_data_step']:
        test_data.step = config['test_data_step']

    #import pdb; pdb.set_trace()

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    model_state_dict = model.state_dict()

    '''
    #bpr_load_state_dict = torch.load('/home/d1/peijie/task/RecBole/archive/ml-1m/model/BPR-ml-1m-Dec-24-2021_14-48-18.pth')['state_dict']
    bpr_load_state_dict = torch.load('/home/d1/peijie/task/RecBole/archive/ml-100k/model/BPR-ml-100k-Dec-24-2021_14-40-39.pth')['state_dict']
    #bpr_load_state_dict = torch.load('/home/d1/peijie/task/RecBole/archive/pinterest/model/BPR-pinterest-Dec-24-2021_15-06-27.pth')['state_dict']
    #bpr_load_state_dict = torch.load('/home/d1/peijie/task/RecBole/archive/yelp/model/BPR-yelp-Dec-01-2021_16-07-06.pth')['state_dict']
    for key, value in bpr_load_state_dict.items():
        if key in model_state_dict:
            model_state_dict[key] = value
    #'''

    '''
    cluster_load_state_dict = torch.load('/home/d1/peijie/task/RecBole/saved/SGL_MY-ml-100k-Jan-01-2022_18-47-22.pth')['state_dict']
    for key, value in cluster_load_state_dict.items():
        if 'cluster' in key:
            model_state_dict[key] = value
    #'''

    #model_state_dict = torch.load('/home/d1/peijie/task/RecBole/archive/ml-1m/model/SGL_MY-ml-1m-Jan-02-2022_18-39-06.pth')['state_dict']
    #model_state_dict = torch.load('/home/d1/peijie/task/RecBole/saved/SGL_MY-amazon_office_products-Jan-07-2022_19-25-37.pth')['state_dict']
    #model_state_dict = torch.load('/home/d1/peijie/task/RecBole/saved/SGL_MY-gowalla-Feb-10-2022_17-56-35.pth')['state_dict']
    #import pdb; pdb.set_trace()

    #model.load_state_dict(model_state_dict)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def run_evaluate(dataloader_file=None, model_file=None):
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_logger(config)

    train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_other_parameter(checkpoint.get('other_parameter'))

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # collect information from train_data
    valid_data.step = 256
    test_data.step = 256

    '''
    import pandas as pd
    df=pd.read_csv('/home/d1/peijie/file/task/RecBole/dataset/ml-100k/ml-100k.inter', sep='\t')
    rating_dict = {}
    for index, row in df.iterrows():
        user_token = row['user_id:token']
        item_token = row['item_id:token'] 
        rating = row['rating:float']
        rating_dict[(user_token, item_token)] = rating

    from collections import defaultdict
    user_historical_items_dict, item_historical_users_dict = defaultdict(set), defaultdict(set)
    inter_user_list, inter_item_list = train_data.dataset.inter_feat['user_id'].tolist(), train_data.dataset.inter_feat['item_id'].tolist()
    for idx, user in enumerate(inter_user_list):
        item = inter_item_list[idx]

        user_token = train_data.dataset.field2id_token['user_id'][user]
        item_token = train_data.dataset.field2id_token['item_id'][item]
        
        user_historical_items_dict[user].add('user:%s' % user_token)
        user_historical_items_dict[user].add('%s:%d' % (item_token, rating_dict[(int(user_token), int(item_token))]))
        item_historical_users_dict[item_token].add('%s:%d' % (user_token, rating_dict[(int(user_token), int(item_token))]))
    item_stat_list = dict()
    for item_token, clicks_list in item_historical_users_dict.items():
        item_stat_list[item_token] = 'toal_len: %d' % len(clicks_list)
    user_stat_list = dict()
    for user, clicked_list in user_historical_items_dict.items():
        user_stat_list[user] = 'toal_len: %d' % len(clicked_list)

    user_clicked_items_dict = defaultdict(set)
    inter_user_list, inter_item_list = test_data.dataset.inter_feat['user_id'].tolist(), test_data.dataset.inter_feat['item_id'].tolist()
    for idx, user in enumerate(inter_user_list):
        item = inter_item_list[idx]
        user_clicked_items_dict[user].add(item)

    # model evaluation
    test_result, recommend_user_items_dict, stat_user_items_tensor = trainer.evaluate(rating_dict, test_data, load_best_model=False, \
        show_progress=config['show_progress'], user_clicked_items=user_clicked_items_dict)
    
    import pdb; pdb.set_trace()
    print(test_result)
    #'''

    val_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=config['show_progress'])
    print(val_result)

    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    print(test_result)


def run_sparsity_analysis(dataloader_file=None, model_file=None):
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_logger(config)

    config['groups_num'] = 10
    config['customize'] = True
    config['ssl_strategy'] = 0
    config['lightgcn_flag'] = True
    config['augmentation'] = False

    train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_other_parameter(checkpoint.get('other_parameter'))

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # collect information from train_data
    valid_data.step = 256
    test_data.step = 256

    user_clicked_items_dict = defaultdict(set)
    inter_user_list, inter_item_list = test_data.dataset.inter_feat['user_id'].tolist(), test_data.dataset.inter_feat['item_id'].tolist()
    for idx, user in enumerate(inter_user_list):
        item = inter_item_list[idx]
        user_clicked_items_dict[user].add(item)

    remap_user_dict = dict()
    remap_user_list = sorted(user_clicked_items_dict.keys())
    for remap_idx, user in enumerate(remap_user_list):
        remap_user_dict[user] = remap_idx

    user_clicked_items_num_list = [len(clicked_items_list) for clicked_items_list in user_clicked_items_dict.values()]
    sorted_user_list = np.argsort(user_clicked_items_num_list[1:])

    user_group_dict = defaultdict(list)
    group_flag_list = [2, 4, 8, 16, 64]

    for user, clicked_items_list in user_clicked_items_dict.items():
        for group_idx, group_flag in enumerate(group_flag_list):
            if len(clicked_items_list) <= group_flag:
                user_group_dict[group_idx].append(user)
                break
    
    # model evaluation
    #test_result, recommend_user_items_dict, stat_user_items_tensor = trainer.evaluate(rating_dict, test_data, load_best_model=False, \
    #    show_progress=config['show_progress'], user_clicked_items=user_clicked_items_dict)
    
    #import pdb; pdb.set_trace()
    #print(test_result)

    #val_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=config['show_progress'])
    #print(val_result)

    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'], customize=config['customize'])
    
    final_group_rating_dict = {}
    final_group_sparsity_dict = {}

    if config['customize']:
        validate_list = []
        
        #'''
        for group_idx in range(len(user_group_dict.keys())):
            group_user_list = user_group_dict[group_idx]
            group_list = []
            for user in group_user_list:
                user = remap_user_dict[user]
                group_list.append(test_result[user])
                validate_list.append(test_result[user])
            final_group_rating_dict[group_idx] = np.mean(group_list)
        '''

        block_size = int(len(sorted_user_list) / config['groups_num']) * 1
        for group_idx in range(config['groups_num']):
            group_list = []
            start_idx = group_idx * block_size
            end_idx = min((group_idx + 1) * block_size, len(sorted_user_list))
            for user in sorted_user_list[start_idx:end_idx]:
                user = user - 1
                group_list.append(test_result[user])
                validate_list.append(test_result[user])
            final_group_list.append(np.mean(group_list))
        '''
        #import pdb; pdb.set_trace()

        #for value in final_group_list:
        #    print('%.4f' % value)
    else:
        print(test_result)

    
    # print sparsity for different group of users
    train_user_clicked_items_dict = defaultdict(set)
    inter_user_list, inter_item_list = train_data.dataset.inter_feat['user_id'].tolist(), train_data.dataset.inter_feat['item_id'].tolist()
    for idx, user in enumerate(inter_user_list):
        item = inter_item_list[idx]
        train_user_clicked_items_dict[user].add(item)

    for group_idx in range(len(user_group_dict.keys())):
        group_user_list = user_group_dict[group_idx]
        gtoup_total_interactions = 0
        for user in group_user_list:
            gtoup_total_interactions += len(train_user_clicked_items_dict[user])
        final_group_sparsity_dict[group_idx] = gtoup_total_interactions / len(group_user_list) / max(inter_item_list)

    for group_idx in range(len(user_group_dict.keys())):
        print('sparsity:%.4f' % final_group_sparsity_dict[group_idx])
        print('ndcg:%.4f' % final_group_rating_dict[group_idx])

def objective_function(config_dict=None, config_file_list=None, saved=True, dataloader_file=None, suffix=None):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)

    #dataset = create_dataset(config)
    #train_data, valid_data, test_data = data_preparation(config, dataset)
    
    train_data, valid_data, test_data = load_split_dataloaders(config['dataloader_file'])

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    if suffix:
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model, suffix)
    else:
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file, dataset_file=None, dataloader_file=None):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.
        dataset_file (str, optional): The path of filtered dataset. Defaults to ``None``.
        dataloader_file (str, optional): The path of split dataloaders. Defaults to ``None``.

    Note:
        The :attr:`dataset` will be loaded or created according to the following strategy:
        If :attr:`dataset_file` is not ``None``, the :attr:`dataset` will be loaded from :attr:`dataset_file`.
        If :attr:`dataset_file` is ``None`` and :attr:`dataloader_file` is ``None``,
        the :attr:`dataset` will be created according to :attr:`config`.
        If :attr:`dataset_file` is ``None`` and :attr:`dataloader_file` is not ``None``,
        the :attr:`dataset` will neither be loaded or created.

        The :attr:`dataloader` will be loaded or created according to the following strategy:
        If :attr:`dataloader_file` is not ``None``, the :attr:`dataloader` will be loaded from :attr:`dataloader_file`.
        If :attr:`dataloader_file` is ``None``, the :attr:`dataloader` will be created according to :attr:`config`.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_logger(config)

    dataset = None
    if dataset_file:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)

    if dataloader_file:
        train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)
    else:
        if dataset is None:
            dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
