[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-enhanced-supervised-contrastive/collaborative-filtering-on-yelp2018)](https://paperswithcode.com/sota/collaborative-filtering-on-yelp2018?p=neighborhood-enhanced-supervised-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-enhanced-supervised-contrastive/recommendation-systems-on-gowalla)](https://paperswithcode.com/sota/recommendation-systems-on-gowalla?p=neighborhood-enhanced-supervised-contrastive)



1. The code repository for the paper:  **Peijie Sun** , Le Wu, Kun Zhang, Xiangzhi Chen, Meng Wang.  **Neighborhood-Enhanced Supervised Contrastive Learning for Collaborative Filtering**  (Accepted by TKDE).

2. The dataset can refer to following links([Baidu Netdisk](https://pan.baidu.com/s/1HXFrGavcvGzHzbkIQP_v3w?pwd=ct9x), [Google Drive](https://drive.google.com/drive/folders/1coHwFat2b4prNPQ4Q8QHznbg8rsi6-P1?usp=sharing)). 

3. The parameters files locate in config/amazon-book \ gowalla \ yelp2018 directories

4. As we have updated the proposed model name to NESCL, its previous name is SUPCCL, it can be found in the path recbole/model/general_recommender/supccl.py

5. To train the model, you should first prepare the training environment
- `pip install -r requirements.txt`
- `python setup.py build_ext --inplace` (We adopt the C++ evaluator in https://github.com/kuandeng/LightGCN)

6. Then, you can execute following commands to train the model based on different datasets:

- `python run_recbole_autodl.py --model=SUPCCL --dataset=yelp2018 --config=True --dataloader_file=/root/autodl-fs/yelp2018-for-SUPCCL-dataloader.pth`

- `python run_recbole_autodl.py --model=SUPCCL --dataset=amazon-book --config=True --dataloader_file=/root/autodl-fs/amazon-book-for-SUPCCL-dataloader.pth`

- `python run_recbole_autodl.py --model=SUPCCL --dataset=gowalla --config=True --dataloader_file=/root/autodl-fs/gowalla-for-SUPCCL-dataloader.pth
`

7. The generated log files saved in `log` directory, and the temporal model parameters can saved in the `saved` directory. 

If you are interested in my work, you can also pay attention to my personal website: https://www.peijiesun.com
