from recbole.data import load_split_dataloaders
from scipy import linalg
from time import time
import numpy as np
from numpy.random import default_rng


def svd_x():
    '''
    dataset = 'pinterest'
    dataloader_file = '/home/d1/peijie/task/RecBole/saved_data/pinterest-for-BPR-dataloader.pth'

    train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)

    interaction_matrix = train_data.dataset.inter_matrix(form='coo').astype(np.float32)
    #'''


    rng = default_rng()
    m, n = 90000, 60000
    interaction_matrix = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))
    
    t1 = time()
    U, s, Vh = linalg.svd(interaction_matrix)
    t2 = time()

    print('cost time:%.4fs' % (t2 - t1))
    import pdb; pdb.set_trace()