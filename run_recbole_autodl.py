# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse, os

from recbole.quick_start import run_recbole, run_evaluate
from recbole.utils import notify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config', type=bool, default=None, help='whether to set config files')
    parser.add_argument('--dataloader_file', type=str, default=None, help='saved dataloader_file')
    parser.add_argument('--model_file', type=str, default=None, help='saved model_file')
    parser.add_argument('--evaluate', type=bool, default=False, help='switch the training & evaluate')

    args, _ = parser.parse_known_args()

    #config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    if args.config:
        config_file_list = []
        config_file_list.append('config/%s/%s.yaml' % (args.dataset, args.dataset))
        if os.path.exists('config/%s/%s-%s.yaml' % (args.dataset, args.model.lower(), args.dataset)):
            config_file_list.append('config/%s/%s-%s.yaml' % (args.dataset, args.model.lower(), args.dataset))
    else:
        config_file_list = None
    
    #import pdb; pdb.set_trace()

    #'''
    if not args.evaluate:
        run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, dataloader_file=args.dataloader_file)
    else:
        run_evaluate(dataloader_file=args.dataloader_file, model_file=args.model_file)
    #'''

    title = 'AutoDL-%s-%s-training' % (args.model, args.dataset)
    content = 'AutoDL-success'
    notify(title, content)
    os.system('shutdown')