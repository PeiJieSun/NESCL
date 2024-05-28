from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
from recbole.config import Config
from recbole.trainer.trainer import Trainer_Fusion

import numpy as np

def x1():
    dataset = 'pinterest'
    dataloader_file = '/home/d1/peijie/task/RecBole/saved_data/pinterest-for-BPR-dataloader.pth'
    config_file_list = []
    config_file_list.append('/home/d1/peijie/task/RecBole/config/%s/%s.yaml' % (dataset, dataset))

    train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

    config_1 = Config(model='LightGCN', dataset=dataset, config_file_list=config_file_list, config_dict=None)

    if train_data.neg_sample_num !=  config_1['train_neg_sample_args']['by']:
        train_data.neg_sample_num = config_1['train_neg_sample_args']['by']
    valid_data.step = 256
    test_data.step = 256

    #'''
    
    model_1 = get_model(config_1['model'])(config_1, train_data.dataset).to(config_1['device'])
    trainer_1 = get_trainer(config_1['MODEL_TYPE'], config_1['model'])(config_1, model_1)
    model_file_1 = '/home/d1/peijie/task/RecBole/archive/pinterest/model/LightGCN-pinterest-Nov-08-2021_10-52-25.pth'
    valid_ndcg_users_1 = trainer_1.evaluate_X(valid_data, load_best_model=True, model_file=model_file_1)
    test_ndcg_users_1 = trainer_1.evaluate_X(test_data, load_best_model=False, model_file=model_file_1)
    #'''
    #import pdb; pdb.set_trace()

    #'''
    config_2 = Config(model='ItemKNN', dataset=dataset, config_file_list=config_file_list, config_dict=None)
    model_2 = get_model(config_2['model'])(config_2, train_data.dataset).to(config_2['device'])
    trainer_2 = get_trainer(config_2['MODEL_TYPE'], config_2['model'])(config_2, model_2)
    model_file_1 = '/home/d1/peijie/task/RecBole/archive/pinterest/model/ItemKNN-pinterest-Dec-16-2021_22-01-23.pth'
    valid_ndcg_users_2 = trainer_2.evaluate_X(valid_data, load_best_model=True, model_file=model_file_1)
    test_ndcg_users_2 = trainer_2.evaluate_X(test_data, load_best_model=False, model_file=model_file_1)
    #'''


    '''
    config_2 = Config(model='BPR', dataset=dataset, config_file_list=config_file_list, config_dict=None)
    model_2 = get_model(config_2['model'])(config_2, train_data.dataset).to(config_2['device'])
    trainer_2 = get_trainer(config_2['MODEL_TYPE'], config_2['model'])(config_2, model_2)
    model_file_2 = '/home/d1/peijie/task/RecBole/archive/pinterest/model/BPR-pinterest-Nov-05-2021_14-30-12.pth'
    valid_ndcg_users_2 = trainer_2.evaluate_X(valid_data, load_best_model=True, model_file=model_file_2)
    test_ndcg_users_2 = trainer_2.evaluate_X(test_data, load_best_model=False, model_file=model_file_2)
    #'''

    xx_users = {}
    # users ndcg = 0; user: 2, 3

    print('user, item with ndcg == 0')
    xx_users = item_list(2, xx_users, train_data)
    xx_users = item_list(3, xx_users, train_data)
    xx_users = item_list(4, xx_users, train_data)
    xx_users = item_list(6, xx_users, train_data)
    xx_users = item_list(7, xx_users, train_data)
    xx_users = item_list(10, xx_users, train_data)

    # users ndcg 5
    print('ItemKNN > LightGCN')
    xx_users = item_list(5, xx_users, train_data)
    xx_users = item_list(18, xx_users, train_data)
    xx_users = item_list(21, xx_users, train_data)
    xx_users = item_list(22, xx_users, train_data)

    # users ndcg 1
    print('ItemKNN < LightGCN')
    xx_users = item_list(1, xx_users, train_data)
    xx_users = item_list(9, xx_users, train_data)
    xx_users = item_list(20, xx_users, train_data)
    xx_users = item_list(23, xx_users, train_data)
    import pdb; pdb.set_trace()

    

    trainer_fusion = Trainer_Fusion(config_1, config_2, model_1, model_2)
    trainer_fusion.evaluate_X(valid_data, load_best_model=True, model_file_1=model_file_1, model_file_2=model_file_2,
        valid_users_confidence_1=valid_ndcg_users_1, valid_users_confidence_2=valid_ndcg_users_2)

    import pdb; pdb.set_trace()

def item_list(user_id, xx_users, train_data):
    xx_users[user_id] = [(train_data.dataset['item_id']==item).sum().item() for item in train_data.dataset['item_id'][(train_data.dataset['user_id']==user_id)]]
    print('user_id:%d' % user_id)
    print(xx_users[user_id])
    print(np.mean(xx_users[user_id]))
    print(np.std(xx_users[user_id]))
    return xx_users