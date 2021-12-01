from ogb.io import DatasetSaver
import numpy as np
import networkx as nx
import os
import torch
# constructor
dataset_name = 'ogbg-toy'
saver = DatasetSaver(dataset_name = dataset_name, is_hetero = False, version = 1)

num_data=10000

split_idx = dict()
permTrainVal = np.random.permutation(9000)
permTest = np.random.permutation(np.arange(9000,10000))
split_idx['train'] = permTrainVal[:int(0.8*num_data)]
split_idx['valid'] = permTrainVal[int(0.8*num_data): int(0.9*num_data)]
split_idx['test'] = permTest[int(0.9*num_data):]
#saver.save_split(split_idx, split_name = 'random'
torch.save(split_idx, '../split_dict.pt')
