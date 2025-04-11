import math
import importlib
import torch
from utils.model_utils import build_mlp, Gen_Index, Aggre_Index, MLP
from model.models.baseline import BaseNetwork
import inspect
from collections import OrderedDict
import os
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from copy import deepcopy
from torch_scatter import scatter
from inspect import signature

def filter_args(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function
    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering
    Returns
    -------
    filtered : dict
        Dictionary containing only keys that are arguments of func
    """
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered


def filter_args_create(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function
    and creates a function with those arguments
    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering
    Returns
    -------
    func : Function
        Function with filtered keys as arguments
    """
    return func(**filter_args(func, keys))

class GeometricEnhancer(nn.Module):
    def __init__(self, dim_node: int, dim_edge: int, dim_geo: int = 11):
        super().__init__()
        self.node_enhancer = build_mlp([dim_node, dim_node * 2, dim_node], do_bn=True)
        self.edge_enhancer = build_mlp([dim_edge + dim_geo, dim_edge * 2, dim_edge], do_bn=True)
        
    def forward(self, node_features, edge_features, geo_features):
        enhanced_nodes = self.node_enhancer(node_features)
        edge_with_geo = torch.cat([edge_features, geo_features], dim=1)
        enhanced_edges = self.edge_enhancer(edge_with_geo)
        return enhanced_nodes, enhanced_edges

class MSG_FAN_EDGE_UPDATE(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='max',
                 attn_dropout: float = 0.5,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])
        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])

        self.combined_mlp = build_mlp([dim_node*2, dim_edge], do_bn=use_bn, on_last=False)
        
        self.cross_att1_q = build_mlp([dim_edge, dim_edge])
        self.cross_att1_k = build_mlp([dim_edge, dim_edge])
        self.cross_att1_v = build_mlp([dim_edge, dim_edge])
        
        self.cross_att2_q = build_mlp([dim_edge, dim_edge])
        self.cross_att2_k = build_mlp([dim_edge, dim_edge])
        self.cross_att2_v = build_mlp([dim_edge, dim_edge])
        
        self.edge_update_mlp = build_mlp([dim_edge*3, dim_edge*2, dim_edge], 
                                         do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        return self.propagate(edge_index, x=x, edge_feature=edge_feature, x_ori=x)

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        num_node = x_i.size(0)
        
        combined = self.combined_mlp(torch.cat([x_i, x_j], dim=1))
        
        q1 = self.cross_att1_q(combined)
        k1 = self.cross_att1_k(edge_feature)
        v1 = self.cross_att1_v(edge_feature)
        
        att1_scores = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(q1.size(-1))
        att1_probs = torch.nn.functional.softmax(att1_scores, dim=-1)
        att1_probs = self.dropout(att1_probs)
        cross_att1_output = torch.matmul(att1_probs, v1)
        
        q2 = self.cross_att2_q(edge_feature)
        k2 = self.cross_att2_k(combined)
        v2 = self.cross_att2_v(combined)
        
        att2_scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(q2.size(-1))
        att2_probs = torch.nn.functional.softmax(att2_scores, dim=-1)
        att2_probs = self.dropout(att2_probs)
        cross_att2_output = torch.matmul(att2_probs, v2)
        
        updated_edge = self.edge_update_mlp(
            torch.cat([edge_feature, cross_att1_output, cross_att2_output], dim=1))
        
        x_i = self.proj_q(x_i).view(
            num_node, self.dim_node_proj, self.num_head)  # [N,D,H]
        edge = self.proj_k(edge_feature).view(
            num_node, self.dim_edge_proj, self.num_head)  # [M,D,H]
        x_j = self.proj_v(x_j)
        
        att = self.nn_att(torch.cat([x_i, edge], dim=1))  # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j

        return [value, updated_edge, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                            dim_size=dim_size, reduce=self.aggr)
        return inputs

    def update(self, x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori, x[0]], dim=1))
        return x
    
class MSG_FAN_Masking(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='max',
                 attn_dropout: float = 0.5,
                 node_mask_prob: float = 0.3,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)
        
        self.node_mask_prob = node_mask_prob

        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])

        self.nn_edge = build_mlp([dim_node*2+dim_edge, (dim_node+dim_edge), dim_edge],
                                 do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        if edge_feature.shape[0] != edge_index.shape[1]:
            print(f"Warning: edge_feature shape {edge_feature.shape} doesn't match edge_index shape {edge_index.shape}")
            min_edges = min(edge_feature.shape[0], edge_index.shape[1])
            edge_feature = edge_feature[:min_edges]
            edge_index = edge_index[:, :min_edges]
        
        if self.training:
            return self.propagate_with_masking(edge_index, x=x, edge_feature=edge_feature, x_ori=x)
        else:
            return self.propagate_without_masking(edge_index, x=x, edge_feature=edge_feature, x_ori=x)
    
    def propagate_with_masking(self, edge_index, **kwargs):
        x = kwargs.get('x')
        edge_feature = kwargs.get('edge_feature')
        x_ori = kwargs.get('x_ori')
        
        num_nodes = x.size(0)
        node_mask = torch.rand(num_nodes, device=edge_index.device) >= self.node_mask_prob
        
        masked_x = x.clone()
        masked_x[~node_mask] = 0 
        kwargs['x'] = masked_x
        
        result = super().propagate(edge_index, **kwargs)
        
        return result[0], result[1], edge_index, result[2]
    
    def propagate_without_masking(self, edge_index, **kwargs):
        result = super().propagate(edge_index, **kwargs)
        return result[0], result[1], result[2]

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        '''
        x_i [N, D_N]
        x_j [N, D_N]
        '''
        num_node = x_i.size(0)

        '''triplet'''
        triplet_feature = torch.cat([x_i, edge_feature, x_j], dim=1)
        triplet_feature = self.nn_edge(triplet_feature)

        '''FAN'''
        # proj
        x_i = self.proj_q(x_i).view(
            num_node, self.dim_node_proj, self.num_head)  # [N,D,H]
        edge = self.proj_k(edge_feature).view(
            num_node, self.dim_edge_proj, self.num_head)  # [M,D,H]
        x_j = self.proj_v(x_j)
        # est attention
        att = self.nn_att(torch.cat([x_i, edge], dim=1))  # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j

        return [value, triplet_feature, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                            dim_size=dim_size, reduce=self.aggr)
        return inputs

    def update(self, x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori, x[0]], dim=1))
        return x

class BidirectionalEdgeLayer(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool = True,
                 aggr='max',
                 attn_dropout: float = 0.3,
                 flow: str = 'target_to_source',
                 use_distance_mask: bool = True,
                 use_node_attention: bool = False):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.use_distance_mask = use_distance_mask
        self.use_node_attention = use_node_attention

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_v = build_mlp([dim_node, dim_atten])
        
        self.proj_k = build_mlp([dim_edge, dim_edge])
        
        self.distance_mlp = build_mlp([4, 32, 1], do_bn=use_bn, on_last=False)
        
        if self.use_node_attention:
            self.node_distance_mlp = build_mlp([1, 32, 1], do_bn=False, on_last=False)
            self.node_self_attn_q = build_mlp([dim_node, dim_node], do_bn=use_bn)
            self.node_self_attn_k = build_mlp([dim_node, dim_node], do_bn=use_bn)
            self.node_self_attn_v = build_mlp([dim_node, dim_node], do_bn=use_bn)
        
        self.nn_edge_update = build_mlp([dim_node*2+dim_edge*2, dim_node+dim_edge*2, dim_edge],
                                       do_bn=use_bn, on_last=False)
        
        self.edge_attention_mlp = build_mlp([dim_edge*2, dim_edge], do_bn=use_bn, on_last=False)
        
        self.nn_node_update = build_mlp([dim_node+dim_edge, dim_node+dim_edge, dim_node],
                                       do_bn=use_bn, on_last=False)
        
        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, 
                          self.dim_node_proj+self.dim_edge_proj,
                          self.dim_edge_proj])
        
        self.node_nonlinear_mlp = build_mlp([dim_node, dim_node, dim_node], 
                                       do_bn=use_bn, on_last=False)
        
        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()
        
        self.sigmoid = torch.nn.Sigmoid()

    def create_node_distance_mask(self, node_positions):

        if not self.use_node_attention:
            return None
            
        num_nodes = node_positions.size(0)
        distance_matrix = torch.zeros((num_nodes, num_nodes), device=node_positions.device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                diff = node_positions[i] - node_positions[j]
                distance_matrix[i, j] = torch.norm(diff, p=2)
        
        input_tensor = distance_matrix.view(-1, 1)
        output = self.node_distance_mlp(input_tensor)
        
        attention_mask = self.sigmoid(output).view(num_nodes, num_nodes)
        return attention_mask
    
    def apply_node_self_attention(self, x, distance_mask=None):

        if not self.use_node_attention:
            return x
            
        num_nodes = x.size(0)
        
        q = self.node_self_attn_q(x)  # [N, D]
        k = self.node_self_attn_k(x)  # [N, D]
        v = self.node_self_attn_v(x)  # [N, D]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_node)  # [N, N]
        
        if distance_mask is not None:
            attn_scores = attn_scores * distance_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, N]
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)  # [N, D]
        
        return output

    def forward(self, x, edge_feature, edge_index, node_positions=None, p_mask=0.0):
        row, col = edge_index
        
        edge_id_mapping = {}
        for idx, (i, j) in enumerate(zip(row, col)):
            edge_id_mapping[(i.item(), j.item())] = idx
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        
        for idx, (i, j) in enumerate(zip(row, col)):
            if (j.item(), i.item()) in edge_id_mapping:
                reverse_idx = edge_id_mapping[(j.item(), i.item())]
                reverse_edge_feature[idx] = edge_feature[reverse_idx]
        
        distance_mask = None
        if self.use_distance_mask and node_positions is not None:
            distance_features = []
            for i, j in zip(row, col):
                i, j = i.item(), j.item()
                pos_i, pos_j = node_positions[i], node_positions[j]
                diff = pos_i - pos_j
                dist = torch.norm(diff, p=2)
                distance_features.append(torch.cat([diff, dist.unsqueeze(0)], dim=0))
            
            distance_features = torch.stack(distance_features)
            
            distance_mask = self.distance_mlp(distance_features)
            distance_mask = self.sigmoid(distance_mask).squeeze(-1)
        
        outgoing_edges = {}
        incoming_edges = {}
        
        for idx, (i, j) in enumerate(zip(row, col)):
            i, j = i.item(), j.item()
            if i not in outgoing_edges:
                outgoing_edges[i] = []
            outgoing_edges[i].append((idx, j))
            
            if j not in incoming_edges:
                incoming_edges[j] = []
            incoming_edges[j].append((idx, i))
        
        updated_node, updated_edge, prob = self.propagate(
            edge_index, 
            x=x, 
            edge_feature=edge_feature,
            reverse_edge_feature=reverse_edge_feature,
            distance_mask=distance_mask,
            x_ori=x
        )
        
        if self.use_node_attention and self.use_distance_mask and node_positions is not None:
            node_distance_mask = self.create_node_distance_mask(node_positions)
            node_self_attn_output = self.apply_node_self_attention(updated_node, node_distance_mask)
            updated_node = updated_node + node_self_attn_output
        
        twin_edge_attention = torch.zeros((x.size(0), self.dim_edge*2), device=x.device)
        
        if self.training:
            outgoing_mask = (torch.rand(x.size(0)) > p_mask).to(x.device)
            incoming_mask = (torch.rand(x.size(0)) > p_mask).to(x.device)
        else:
            outgoing_mask = torch.ones(x.size(0), device=x.device)
            incoming_mask = torch.ones(x.size(0), device=x.device)
        
        for node_id in range(x.size(0)):
            outgoing_feature = torch.zeros(self.dim_edge, device=x.device)
            if node_id in outgoing_edges:
                for edge_idx, _ in outgoing_edges[node_id]:
                    outgoing_feature += updated_edge[edge_idx]
                if len(outgoing_edges[node_id]) > 0:
                    outgoing_feature /= len(outgoing_edges[node_id])
            outgoing_feature = outgoing_feature * outgoing_mask[node_id]
            
            incoming_feature = torch.zeros(self.dim_edge, device=x.device)
            if node_id in incoming_edges:
                for edge_idx, _ in incoming_edges[node_id]:
                    incoming_feature += updated_edge[edge_idx]
                if len(incoming_edges[node_id]) > 0:
                    incoming_feature /= len(incoming_edges[node_id])
            incoming_feature = incoming_feature * incoming_mask[node_id]
            
            twin_edge_attention[node_id] = torch.cat([outgoing_feature, incoming_feature], dim=0)
        
        edge_attention = self.edge_attention_mlp(twin_edge_attention)
        edge_attention = self.sigmoid(edge_attention)
        
        # node_feature_nonlinear = torch.nn.functional.relu(updated_node)  # f(v_i^l)
        node_feature_nonlinear = self.node_nonlinear_mlp(updated_node)
        final_node = node_feature_nonlinear * edge_attention  # ⊙ β(A_ε)
        
        return final_node, updated_edge, prob

    def message(self, x_i: Tensor, x_j: Tensor, 
                edge_feature: Tensor, reverse_edge_feature: Tensor,
                distance_mask: Optional[Tensor] = None) -> Tensor:
        '''
        x_i: 소스 노드 특징 [N, D_N]
        x_j: 타겟 노드 특징 [N, D_N]
        edge_feature: 정방향 에지 특징 [N, D_E]
        reverse_edge_feature: 역방향 에지 특징 [N, D_E]
        distance_mask: 거리 기반 마스킹 가중치 [N] (선택적)
        '''
        num_edge = x_i.size(0)
        
        updated_edge = self.nn_edge_update(
            torch.cat([x_i, edge_feature, reverse_edge_feature, x_j], dim=1)
        )
        
        x_i_proj = self.proj_q(x_i).view(
            num_edge, self.dim_node_proj, self.num_head)  # [N, D, H]
        edge_proj = self.proj_k(edge_feature).view(
            num_edge, self.dim_edge_proj, self.num_head)  # [N, D, H]
        x_j_val = self.proj_v(x_j)
        
        att = self.nn_att(torch.cat([x_i_proj, edge_proj], dim=1))  # [N, D, H]
        
        if distance_mask is not None:
            distance_mask = distance_mask.view(-1, 1, 1)
            att = att * distance_mask
        
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        
        weighted_value = prob.reshape_as(x_j_val) * x_j_val
        
        return [weighted_value, updated_edge, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        weighted_value, updated_edge, prob = inputs
        weighted_value = scatter(weighted_value, index, dim=self.node_dim,
                                dim_size=dim_size, reduce=self.aggr)
        return weighted_value, updated_edge, prob

    def update(self, inputs, x_ori):
        weighted_value, updated_edge, prob = inputs
        
        updated_node = self.nn_node_update(
            torch.cat([x_ori, weighted_value], dim=1)
        )
        
        return updated_node, updated_edge, prob
    
class GraphEdgeAttenNetworkLayers_masking(torch.nn.Module):
    """ A sequence of scene graph convolution layers with node masking only """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        node_mask_prob = kwargs.get('node_mask_prob', 0.3)
        
        for _ in range(self.num_layers):
            self.gconvs.append(MSG_FAN_Masking(
                dim_node=kwargs['dim_node'],
                dim_edge=kwargs['dim_edge'],
                dim_atten=kwargs['dim_atten'],
                num_heads=kwargs['num_heads'],
                use_bn=kwargs['use_bn'],
                aggr=kwargs['aggr'],
                attn_dropout=kwargs.get('attn_dropout', 0.1),
                node_mask_prob=node_mask_prob,
                flow=kwargs.get('flow', 'target_to_source')
            ))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            
            if self.training:
                # Training with node masking
                node_feature, edge_feature, edges_indices, prob = gconv(
                    node_feature, edge_feature, edges_indices)
            else:
                # Inference without masking
                node_feature, edge_feature, prob = gconv.propagate_without_masking(
                    edges_indices, x=node_feature, edge_feature=edge_feature, x_ori=node_feature)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        
        return node_feature, edge_feature, probs

class BidirectionalEdgeGraphNetwork(torch.nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']
        self.use_distance_mask = kwargs.get('use_distance_mask', True)
        self.use_node_attention = kwargs.get('use_node_attention', False)
        self.edge_mask_prob = kwargs.get('edge_mask_prob', 0.0)
        self.use_geometric_enhancer = kwargs.get('use_geometric_enhancer', False)
        self.use_lambda_control = kwargs.get('use_lambda_control', True)
        
        if self.use_lambda_control:
            self.node_lambdas = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(0.0))
                for i in range(self.num_layers)
            ])
            
            # Edge
            # self.edge_lambdas = torch.nn.ParameterList([
            #     torch.nn.Parameter(torch.tensor(0.0))
            #     for i in range(self.num_layers)
            # ])

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        kwargs['use_node_attention'] = self.use_node_attention

        if self.use_geometric_enhancer:
            self.geo_enhancer = GeometricEnhancer(
                dim_node=kwargs['dim_node'],
                dim_edge=kwargs['dim_edge'],
                dim_geo=11
            )
        
        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(BidirectionalEdgeLayer, kwargs))

    def forward(self, node_feature, edge_feature, edges_indices, descriptor=None):
        probs = list()

        node_positions = None
        geo_features = None

        if descriptor is not None and (self.use_distance_mask or self.use_node_attention or self.use_geometric_enhancer):
            node_positions = descriptor[:, :3]
            
            if self.use_geometric_enhancer:
                row, col = edges_indices
                geo_features = []
                for i, j in zip(row, col):
                    i, j = i.item(), j.item()
                    pos_i, pos_j = node_positions[i], node_positions[j]
                    diff = pos_i - pos_j
                    dist = torch.norm(diff, p=2)
                    
                    geo_feat = torch.cat([
                        diff,                     # 위치 차이 (3차원)
                        dist.unsqueeze(0),        # 유클리드 거리 (1차원)
                        diff / (dist + 1e-10),    # 정규화된 방향 (3차원)
                        torch.tensor([
                            diff[0] > 0,          # x 방향 관계 (1차원)
                            diff[1] > 0,          # y 방향 관계 (1차원)
                            diff[2] > 0,          # z 방향 관계 (1차원)
                            dist < torch.median(torch.tensor([dist]))  # 근접성 (1차원)
                        ], device=diff.device).float()
                    ], dim=0)
                    geo_features.append(geo_feat)
                
                geo_features = torch.stack(geo_features)
                
                node_feature, edge_feature = self.geo_enhancer(node_feature, edge_feature, geo_features)
        
        initial_node_feature = node_feature.clone()
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            
            updated_node, updated_edge, prob = gconv(
                node_feature, edge_feature, edges_indices, node_positions, p_mask=self.edge_mask_prob
            )
            
            if self.use_lambda_control:
                node_lambda = torch.sigmoid(self.node_lambdas[i])
                node_feature = node_lambda * node_feature + (1 - node_lambda) * updated_node
                
                edge_feature = updated_edge
                
                # Edge
                # edge_lambda = torch.sigmoid(self.edge_lambdas[i])
                # edge_feature = edge_lambda * edge_feature + (1 - edge_lambda) * updated_edge
            else:
                node_feature = updated_node
                edge_feature = updated_edge

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
                
        return node_feature, edge_feature, probs