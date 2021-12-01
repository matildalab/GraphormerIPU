from collator import collator
from ogb_wrapper import MyPygPCQM4MDataset2

# from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
from functools import partial

dataset = None

def get_dataset(dataset_name = 'PCQM4M-LSC'):
    global dataset
    if dataset is not None:
        return dataset

    if dataset_name == 'PCQM4M-LSC':
        dataset = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': ogb.lsc.PCQM4MEvaluator(),
            'dataset': MyPygPCQM4MDataset2(),
            'max_node': 30,
        }
    else:
        raise NotImplementedError
            
    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset

class GraphDataModule():
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = 'PCQM4M-LSC',
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        rel_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.rel_pos_max = rel_pos_max
        self.seed = seed
        assert 0 <= self.seed <= 7
    
    def setup(self,stage: str=None):
        import numpy as np
        split_idx = self.dataset['dataset'].get_idx_split()
        train_val = np.hstack([split_idx["train"], split_idx["valid"]]).tolist()
        
        self.dataset_train = self.dataset['dataset'][train_val]
        self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
        self.dataset_test = self.dataset['dataset'][split_idx["test"]]

    # def train_dataloader(self):
    #     opts=poptorch.Options()
    #     loader = poptorch.DataLoader(
    #         self.training_opts,
    #         self.dataset_train,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)['max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
    #         # mode=poptorch.DataLoaderMode.Async,
    #     )
    #     print('len(train_dataloader)', len(loader))
    #     return loader

    # def val_dataloader(self):
    #     opts = poptorch.Options()
    #     loader = poptorch.DataLoader(
    #         self.training_opts,
    #         self.dataset_val,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)['max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
    #         # mode=poptorch.DataLoaderMode.Async,
    #     )
    #     print('len(val_dataloader)', len(loader))
    #     return loader

    # def test_dataloader(self):
    #     opts=poptorch.Options()
    #     loader = poptorch.DataLoader(
    #         opts,
    #         self.dataset_test,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=False,
    #         collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)['max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
    #         # mode=poptorch.DataLoaderMode.Async,
    #     )
    #     print('len(test_dataloader)', len(loader))
    #     return loader
