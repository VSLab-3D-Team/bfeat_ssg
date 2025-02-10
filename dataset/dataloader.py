from pydoc import describe
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import numpy as np

class CustomSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter
    
class CustomMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter


class CustomDataLoader(DataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        if worker_init_fn is None:
            worker_init_fn = self.init_fn
        super().__init__(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.config = config
        
    def init_fn(self, worker_id):
        np.random.seed(self.config.seed + worker_id)
        
    def __iter__(self):
        if self.num_workers == 0:
            return CustomSingleProcessDataLoaderIter(self)
        else:
            return CustomMultiProcessingDataLoaderIter(self)


def collate_fn_bfeat(batch):
    obj_point_list, obj_label_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_indices, descriptor = [], []
    batch_ids = []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        rel_point_list.append(b[1])
        descriptor.append(b[2])
        rel_label_list.append(b[3])
        obj_label_list.append(b[4])
        edge_indices.append(b[5] + count)
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))

    return torch.cat(obj_point_list, dim=0), \
        torch.cat(rel_point_list, dim=0), \
        torch.cat(descriptor, dim=0), \
        torch.cat(rel_label_list, dim=0), \
        torch.cat(obj_label_list, dim=0), \
        torch.cat(edge_indices, dim=0), \
        torch.cat(batch_ids, dim=0)

def collate_fn_bfeat_mv(batch):
    obj_point_list, obj_label_list = [], []
    obj_feats_list, zero_mask_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_indices, descriptor = [], []
    batch_ids = []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_feats_list.append(b[1])
        rel_point_list.append(b[2])
        descriptor.append(b[3])
        rel_label_list.append(b[4])
        obj_label_list.append(b[5])
        zero_mask_list.append(b[6])
        edge_indices.append(b[7] + count)
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))

    return torch.cat(obj_point_list, dim=0), \
        torch.cat(obj_feats_list, dim=0), \
        torch.cat(rel_point_list, dim=0), \
        torch.cat(descriptor, dim=0), \
        torch.cat(rel_label_list, dim=0), \
        torch.cat(obj_label_list, dim=0), \
        torch.cat(zero_mask_list, dim=0), \
        torch.cat(edge_indices, dim=0), \
        torch.cat(batch_ids, dim=0)

def collate_fn_mmg(batch):
    # batch
    obj_point_list, obj_label_list, obj_2d_feats = [], [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    batch_ids = []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_2d_feats.append(b[1])
        obj_label_list.append(b[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(b[4])
        edge_indices.append(b[5] + count)
        descriptor.append(b[6])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))


    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0)
