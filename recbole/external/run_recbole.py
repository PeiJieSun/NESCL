import logging
from logging import getLogger

import os
import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

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
    
    #import pdb; pdb.set_trace()

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
    
    #train_data.step = 1024
    valid_data.step = 256
    test_data.step = 256

    # model loading and initialization
    if config['model'] == 'SLIM_X':
        model = get_model(config['model'])(config, train_data.dataset)#.to(config['device'])
        model.fit()
        model.evaluate(test_data)
    elif config['model'] == 'UltraGCN':
        from recbole.model.general_recommender.ultragcn import ultragcn_start
        ultragcn_start(config, train_data, test_data)
    elif config['model'] == 'SimpleX':
        pass 
    logger.info(model)