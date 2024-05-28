#'''
from recbole.debug.convert_data import first_stage, second_stage, convert_recbole_data_common_data
from recbole.debug.model_fusion import x1
from recbole.debug.svd import svd_x

dataset = 'gowalla'
data_dir = '/home/d1/peijie/task/LightGCN/Data/%s' % dataset
saved_dir = '/home/d1/peijie/task/RecBole/dataset/%s' % dataset

first_stage(data_dir, saved_dir, dataset)
second_stage(data_dir, saved_dir, dataset)

#convert_recbole_data_common_data('/home/d1/peijie/task/RecBole/saved_data/ml-100k-for-BPR-dataloader.pth')

#svd_x()
#'''