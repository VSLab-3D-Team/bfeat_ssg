from typing import Union
from config.define import *
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from dataset.dataset_3dssg import SSGLWBFeat3D, SSGLWBFeat3DwMultiModal
import numpy as np
from tqdm import tqdm

class SSGImbalanceSampler(Sampler):
    def __init__(self, data_source: Union[SSGLWBFeat3D, SSGLWBFeat3DwMultiModal]):
        super().__init__(data_source)
        
        self.indices = list(range(len(data_source)))
        self.dataset = data_source
        print("Calculating weights for oversampling...")
        self.scan_weights = np.ones(len(self.dataset.scan_data))
        for idx, scan in tqdm(enumerate(self.dataset.scan_data)):
            relationship = scan["rel_json"]
            for r in relationship:
                _, _, pred_id, _ = r
                if pred_id in TAIL_PREDICATE_ID:
                    self.scan_weights[idx] += 1
        self.scan_weights = self.scan_weights / self.scan_weights.sum()
        self.scan_weights = torch.from_numpy(self.scan_weights)

    def __iter__(self):
        return iter(torch.multinomial(self.scan_weights, len(self.scan_weights), replacement=True))

    def __len__(self):
        return len(self.indices)


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

def collate_fn_geo_aux(batch):
    obj_point_list, obj_label_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_indices, descriptor = [], []
    edge_clip_feats, edge_feat_mask = [], []
    batch_ids = []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        rel_point_list.append(b[1])
        descriptor.append(b[2])
        edge_clip_feats.append(b[3])
        rel_label_list.append(b[4])
        obj_label_list.append(b[5])
        edge_indices.append(b[6] + count)
        edge_feat_mask.append(b[7])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))

    return torch.cat(obj_point_list, dim=0), \
        torch.cat(rel_point_list, dim=0), \
        torch.cat(descriptor, dim=0), \
        torch.cat(edge_clip_feats, dim=0), \
        torch.cat(rel_label_list, dim=0), \
        torch.cat(obj_label_list, dim=0), \
        torch.cat(edge_indices, dim=0), \
        torch.cat(edge_feat_mask, dim=0), \
        torch.cat(batch_ids, dim=0)

def collate_fn_bfeat_edge_obj_mv(batch):
    obj_point_list, obj_label_list = [], []
    obj_feats_list, zero_mask_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_clip_feats, edge_feat_mask = [], []
    edge_indices, descriptor = [], []
    batch_ids = []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_feats_list.append(b[1])
        edge_clip_feats.append(b[2])
        rel_point_list.append(b[3])
        descriptor.append(b[4])
        rel_label_list.append(b[5])
        obj_label_list.append(b[6])
        edge_indices.append(b[7] + count)
        zero_mask_list.append(b[8])
        edge_feat_mask.append(b[9])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))

    return torch.cat(obj_point_list, dim=0), \
        torch.cat(obj_feats_list, dim=0), \
        torch.cat(edge_clip_feats, dim=0), \
        torch.cat(rel_point_list, dim=0), \
        torch.cat(descriptor, dim=0), \
        torch.cat(rel_label_list, dim=0), \
        torch.cat(obj_label_list, dim=0), \
        torch.cat(zero_mask_list, dim=0), \
        torch.cat(edge_feat_mask, dim=0), \
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
