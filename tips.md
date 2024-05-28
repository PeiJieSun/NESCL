更换新数据集时，需要采取如下步骤：
1. 确定数据集，数据集选择途径https://recbole.io/dataset_list.html
2. 创建data.yaml到/home/peijie/task/RecBole/config目录下
3. 修改data.yaml文件，设置 save_dataloaders: True，以及注释 dataloader_file: 
4. 执行BPR，保存dataloaders文件，指令可以参考/home/peijie/task/RecBole/QuickStart.md
5. 修改data.yaml文件，注释 save_dataloaders: True，以及解注释 dataloader_file: 
6. 将dataloaders文件共享给9822,9922两个服务器
7. 将数据集的相关信息存储到“C:\Users\sunhf\iCloudDrive\Mac\Task\Processing\20210801 图自监督学习”文件夹内
8. 一边实验，一边记录相关信息到相应的表格中。并将生成的log文件和model文件保存到相应的archive文件夹内。