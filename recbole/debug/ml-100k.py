from recbole.data import load_split_dataloaders

dataloader_file = '/home/peijie/task/RecBole/saved_data/ml-100k-for-User_based-dataloader.pth'
train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

import pandas as pd
df=pd.read_csv('/home/d1/peijie/task/RecBole/dataset/ml-100k/ml-100k.inter', sep='\t')
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

import pdb; pdb.set_trace()