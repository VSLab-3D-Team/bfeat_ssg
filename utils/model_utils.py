#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:46:24 2020

@author: sc
"""

import torch
from torch_geometric.nn.conv import MessagePassing
from model.models.baseline import mySequential
import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def MLP(channels: list, do_bn=False, on_last=False, drop_out=None):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-offset):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
            
            if drop_out is not None:
                layers.append(torch.nn.Dropout(drop_out))
    return mySequential(*layers)


def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
     dim_in, dim_out = dim_list[i], dim_list[i + 1]
     layers.append(torch.nn.Linear(dim_in, dim_out))
     final_layer = (i == len(dim_list) - 2)
     if not final_layer or on_last:
       if do_bn:
         layers.append(torch.nn.BatchNorm1d(dim_out))
       if activation == 'relu':
         layers.append(torch.nn.ReLU())
       elif activation == 'leakyrelu':
         layers.append(torch.nn.LeakyReLU())
     if dropout > 0:
       layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)


class TFIDFMaskLayer(object):
    """
    This Module gets single batch of 3D Scene and calculates the attention mask w/ TF-IDF
    Making rare classes more dominant in training scheme
    """
    def __init__(self, num_classes, device):
        super(TFIDFMaskLayer, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.idf_values = torch.zeros(num_classes)
        
    # Batch 안의 3D Scene 안에 object label c_i가 있는가?
    ## batch_id를 통해서 같은 graph 안에 있는지 확인해야함.
    ## 같은 graph 안에 있는 놈들 끼리 weight를 묶어야함.
    def get_mask(
        self, 
        gt_label: torch.Tensor, 
        batch_ids: torch.Tensor
    ):
        """
        Inputs:
            - gt_label: B X 1, labels
            - batch_ids: B X 1, batch id 0 ~ bsz-1
        Outputs:
            - mask: C X 1, TF-IDF based attention mask
        """
        bsz = batch_ids.max().item() + 1
        class_counts = torch.bincount(gt_label.flatten(), minlength=self.num_classes).float()
        tf = class_counts / bsz 
        
        doc_count = torch.zeros(self.num_classes).to(self.device)
        for b in range(bsz):
            batch_mask = torch.where(batch_ids == b)[0]
            obj_label_batch = gt_label[batch_mask]
            count_batch = torch.bincount(obj_label_batch.flatten(), minlength=self.num_classes)
            contains_class = (count_batch > 0).float()
            doc_count += contains_class
            
        idf = torch.log((bsz + 1) / (1 + doc_count))
        weights = tf * idf
        weights = weights / (weights.sum() + 1e-6)  # 정규화
        return weights.to(self.device)

class TFIDFTripletWeight(object):
    def __init__(self, num_classes_obj, num_classes_pred, device):
        self.device = device
        self.num_classes_obj = num_classes_obj
        self.num_classes_pred = num_classes_pred
    
    def get_mask(
        self,
        gt_obj_label: torch.Tensor,
        gt_rel_label: torch.Tensor,
        edge_indices: torch.Tensor,
        batch_ids: torch.Tensor
    ):
        """
        Things to implement: Multi-Label, Edge-based Batch-wise TF-IDF
        Inputs:
            - gt_label: N_edge X N_cls, multi-labeled GT one-hot vector
            - edge_indices: N_edge X 2, <sub idx, obj idx>
            - batch_idx: N_obj X 1, batch ids
        Outputs:
            - weights: N_edge X 1, Edge-wise attention weight
        """
        bsz = batch_ids.max().item() + 1
        tf = torch.zeros((
            self.num_classes_obj, 
            self.num_classes_pred, 
            self.num_classes_obj
        )).to(self.device)
        df = torch.zeros((
            self.num_classes_obj, 
            self.num_classes_pred, 
            self.num_classes_obj,
            bsz
        )).to(self.device)
        # DF(t) = Number of documents contains certain triplets <subject, predicate, object>
        # IDF(t) = N_b / (1 + D(t))
        # TF(t) = 
        # W(e)^{tf-idf} = \product
        edge_weight = torch.zeros(edge_indices.shape[0]).to(self.device)
        for e_id in range(edge_indices.shape[0]):
            idx_eo = edge_indices[e_id][0]
            idx_os = edge_indices[e_id][1]
            sub_label, obj_label = gt_obj_label[idx_eo], gt_obj_label[idx_os]
            s_b_id, o_b_id = batch_ids[idx_eo], batch_ids[idx_os]
            assert s_b_id == o_b_id, "invalid batch index between sub-obj relation"
            edge_label = gt_rel_label[e_id]
            tf[sub_label, :, obj_label] += edge_label
            _mask_rel = (edge_label == 1)
            df[sub_label, _mask_rel, obj_label, s_b_id] = 1
        df = torch.sum(df, dim=-1)
        idf = torch.log((bsz + 1) / (1 + df))
        
        for e_id in range(edge_indices.shape[0]):
            idx_eo = edge_indices[e_id][0]
            idx_os = edge_indices[e_id][1]
            sub_label, obj_label = gt_obj_label[idx_eo], gt_obj_label[idx_os]
            edge_label = gt_rel_label[e_id]
            _mask_rel = (edge_label == 1)
            edge_weight[e_id] = (tf[sub_label, _mask_rel, obj_label] * idf[sub_label, _mask_rel, obj_label]).sum()
        
        edge_weight = edge_weight / (1e-6 + edge_weight.sum()) # N_edge X 1
        return edge_weight

class Gen_Index(MessagePassing):
    """ A sequence of scene graph convolution layers  """
    def __init__(self,flow="target_to_source"):
        super().__init__(flow=flow)
        
    def forward(self, x, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(self.__user_args__,edges_indices,size, {"x":x})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        x_i, x_j = self.message(**msg_kwargs)
        return x_i, x_j
    def message(self, x_i, x_j):
        return x_i,x_j

class Aggre_Index(MessagePassing):
    def __init__(self,aggr='add', node_dim=-2,flow="source_to_target"):
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)
    def forward(self, x, edge_index,dim_size):
        size = self.__check_input__(edge_index, None)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size,{})
        coll_dict['dim_size'] = dim_size
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        x = self.aggregate(x, **aggr_kwargs)
        return x

if __name__ == '__main__':
    flow = 'source_to_target'
    # flow = 'target_to_source'
    g = Gen_Index(flow = flow)
    
    edge_index = torch.LongTensor([[0,1,2],
                                  [2,1,0]])
    x = torch.zeros([3,5])
    x[0,:] = 0
    x[1,:] = 1
    x[2,:] = 2
    x_i,x_j = g(x,edge_index)
    print('x_i',x_i)
    print('x_j',x_j)
    
    tmp = torch.zeros_like(x_i)
    tmp = torch.zeros([5,2])
    edge_index = torch.LongTensor([[0,1,2,1,0],
                                  [2,1,1,1,1]])
    for i in range(5):
        tmp[i] = -i
    aggr = Aggre_Index(flow=flow,aggr='max')
    xx = aggr(tmp, edge_index,dim_size=x.shape[0])
    print(x)
    print(xx)