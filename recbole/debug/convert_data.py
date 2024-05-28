import sys
sys.path.append('/home/d1/peijie/task')

from RecBole.recbole.config import Config
from RecBole.recbole.utils import init_seed
from RecBole.recbole.data import create_dataset_x, save_split_dataloaders, data_preparation, load_split_dataloaders
from RecBole.recbole.data.utils import get_dataloader, create_samplers

# First stage, convert the dataset used in other papers to the datasets we can use. 
def first_stage(data_dir, saved_dir, dataset):
    train_path =  '%s/train.txt' % data_dir
    train_inter = '%s/train.%s.inter' % (saved_dir, dataset)

    w1 = open(train_inter, 'w')
    w1.write('user_id:token\titem_id:token\n')

    with open(train_path) as f:
        for line in f:
            tmp = line.split()
            for item in tmp[1:]:
                w1.write('%s\t%s\n' % (tmp[0], item)) 

    w1.flush(); w1.close()

    test_path = '%s/test.txt' % data_dir
    test_inter = '%s/test.%s.inter' % (saved_dir, dataset)

    w2 = open(test_inter, 'w')
    w2.write('user_id:token\titem_id:token\n')

    with open(test_path) as f:
        for line in f:
            tmp = line.split()
            for item in tmp[1:]:
                w2.write('%s\t%s\n' % (tmp[0], item)) 

    w2.flush(); w2.close()

# convert the train data into DataLoader(train_data), and convert the test data into the test.xx.inter
def second_stage(data_dir, saved_dir, dataset):
    model = 'BPR'
    config_file_list = []
    config_file_list.append('/home/d1/peijie/task/RecBole/config/%s/%s.yaml' % (dataset, dataset))

    config = Config(model=model, dataset=dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    config['data_path'] = '%s/train.%s.inter' % (saved_dir, dataset)

    #import pdb; pdb.set_trace()
    train_dataset = create_dataset_x(config)

    #################################################################
    config['data_path'] = '%s/test.%s.inter' % (saved_dir, dataset)
    test_dataset = create_dataset_x(config, pre_flag=1, pre_field2id_token=train_dataset.field2id_token, pre_field2token_id=train_dataset.field2token_id)

    #train_data, valid_data, test_data = data_preparation(config, train_dataset)
    #import pdb; pdb.set_trace()

    train_dataset._change_feat_format()
    test_dataset._change_feat_format()

    built_datasets = [train_dataset, test_dataset, test_dataset]
    train_sampler, valid_sampler, test_sampler = create_samplers(config, train_dataset, built_datasets)

    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)

    save_split_dataloaders(config, dataloaders=(train_data, test_data, test_data))

def convert_recbole_data_common_data(dataloader_file):
    train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

    f2=open('/home/d1/peijie/task/Open-Match-Benchmark/data/ML-100K/train.txt', 'w')
    user_list = train_data.dataset.inter_feat['user_id'].tolist()
    item_list = train_data.dataset.inter_feat['item_id'].tolist()

    for idx, user_id in enumerate(user_list):
        f2.write('%d\t%d\n' % (user_id, item_list[idx]))

    f2=open('/home/d1/peijie/task/Open-Match-Benchmark/data/ML-100K/test.txt', 'w')
    user_list = valid_data.dataset.inter_feat['user_id'].tolist()
    item_list = valid_data.dataset.inter_feat['item_id'].tolist()

    for idx, user_id in enumerate(user_list):
        f2.write('%d\t%d\n' % (user_id, item_list[idx]))

    import pdb; pdb.set_trace()
