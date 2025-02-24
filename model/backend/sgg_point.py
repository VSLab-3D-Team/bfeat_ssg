import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import clip
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv
from torch_scatter import scatter

from model.models.baseline import BaseNetwork
from model.backend.attention import MultiHeadAttention

################################
# Just Copy From SGGpoint Repo #
################################

#####################################################
#                                                   #
#                                                   #
#   Backbone network - PointNet                     #
#                                                   #
#                                                   #
#####################################################

class PointNet(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel, embeddings):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embeddings, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embeddings)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x

#####################################################
#                                                   #
#                                                   #
#   Backbone network - DGCNN (and its components)   #
#                                                   #
#                                                   #
#####################################################

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNN(nn.Module):
    # official DGCNN
    def __init__(self, input_channel, embeddings):
        super(DGCNN, self).__init__()
        self.k = 20
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, embeddings, kernel_size=1, bias=False),nn.BatchNorm1d(embeddings),nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x)
        return x

##################################################
#                                                #
#                                                #
#  Core Network: EdgeGCN                         #
#                                                #
#                                                #
##################################################

class EdgeGCN(torch.nn.Module):
    def __init__(self, num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag):
        super(EdgeGCN, self).__init__()

        self.node_GConv1 = GCNConv(num_node_in_embeddings, num_node_in_embeddings // 2, add_self_loops=True)
        self.node_GConv2 = GCNConv(num_node_in_embeddings // 2, num_node_in_embeddings, add_self_loops=True)

        self.edge_MLP1 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings, num_edge_in_embeddings // 2, 1), nn.ReLU())
        self.edge_MLP2 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings // 2, num_edge_in_embeddings, 1), nn.ReLU())

        self.AttnEdgeFlag = AttnEdgeFlag # boolean (for ablaiton studies)
        self.AttnNodeFlag = AttnNodeFlag # boolean (for ablaiton studies)

        # multi-dimentional (N-Dim) node/edge attn coefficients mappings
        self.edge_attentionND = nn.Linear(num_edge_in_embeddings, num_node_in_embeddings // 2) if self.AttnEdgeFlag else None
        self.node_attentionND = nn.Linear(num_node_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        self.node_indicator_reduction = nn.Linear(num_edge_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

    def concate_NodeIndicator_for_edges(self, node_indicator, batchwise_edge_index):
        node_indicator = node_indicator.squeeze(0)
        
        edge_index_list = batchwise_edge_index.t()
        subject_idx_list = edge_index_list[:, 0]
        object_idx_list = edge_index_list[:, 1]

        subject_indicator = node_indicator[subject_idx_list]  # (num_edges, num_mid_channels)
        object_indicator = node_indicator[object_idx_list]    # (num_edges, num_mid_channels)

        edge_concat = torch.cat((subject_indicator, object_indicator), dim=-1)
        return edge_concat  # (num_edges, num_mid_channels * 2)

    def forward(self, node_feats, edge_feats, edge_index):
        # prepare node_feats & edge_feats in the following formats
        # node_feats: (1, num_nodes,  num_embeddings)
        # edge_feats: (1, num_edges,  num_embeddings)
        # (num_embeddings = num_node_in_embeddings = num_edge_in_embeddings) = 2 * num_mid_channels
        
        #### Deriving Edge Attention
        if self.AttnEdgeFlag:
            edge_indicator = self.edge_attentionND(edge_feats.squeeze(0)).unsqueeze(0).permute(0, 2, 1)  # (1, num_mid_channels, num_edges)
            raw_out_row = scatter(edge_indicator, edge_index.t()[:, 0].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            raw_out_col = scatter(edge_indicator, edge_index.t()[:, 1].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            agg_edge_indicator_logits = raw_out_row * raw_out_col                                        # (1, num_mid_channels, num_nodes)
            agg_edge_indicator = torch.sigmoid(agg_edge_indicator_logits).permute(0, 2, 1).squeeze(0)    # (num_nodes, num_mid_channels)
        else:
            agg_edge_indicator = 1

        #### Node Evolution Stream (NodeGCN)
        node_feats = F.relu(self.node_GConv1(node_feats, edge_index)) * agg_edge_indicator # applying EdgeAttn on Nodes
        node_feats = F.dropout(node_feats, training=self.training)
        node_feats = F.relu(self.node_GConv2(node_feats, edge_index))
        node_feats = node_feats.unsqueeze(0)  # (1, num_nodes, num_embeddings)

        #### Deriving Node Attention
        if self.AttnNodeFlag:
            node_indicator = F.relu(self.node_attentionND(node_feats.squeeze(0)).unsqueeze(0))                  # (1, num_mid_channels, num_nodes)
            agg_node_indicator = self.concate_NodeIndicator_for_edges(node_indicator, edge_index)               # (num_edges, num_mid_channels * 2)
            agg_node_indicator = self.node_indicator_reduction(agg_node_indicator).unsqueeze(0).permute(0,2,1)  # (1, num_mid_channels, num_edges)
            agg_node_indicator = torch.sigmoid(agg_node_indicator)  # (1, num_mid_channels, num_edges)
        else:
            agg_node_indicator = 1

        #### Edge Evolution Stream (EdgeMLP)
        edge_feats = edge_feats.unsqueeze(0).permute(0, 2, 1)                  # (1, num_embeddings, num_edges)
        edge_feats = self.edge_MLP1(edge_feats)                   # (1, num_mid_channels, num_edges)
        edge_feats = F.dropout(edge_feats, training=self.training) * agg_node_indicator    # applying NodeAttn on Edges
        edge_feats = self.edge_MLP2(edge_feats).permute(0, 2, 1)  # (1, num_edges, num_embeddings)

        return node_feats.squeeze(0), edge_feats.squeeze(0)

class EdgeMLP(nn.Module):
    def __init__(self, embeddings, nRelClasses, negative_slope=0.2):
        super(EdgeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.edge_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.edge_linear2 = nn.Linear(mid_channels, nRelClasses, bias=False)

    def forward(self, edge_feats):
        # edge_feats: (1, edges, embeddings)  => edge_logits: (1, edges, nRelClasses)
        x = self.edge_linear1(edge_feats.unsqueeze(0))
        x = self.edge_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(x)
        # we treat it as multi-label classification
        edge_logits = torch.sigmoid(edge_logits)
        return edge_logits.squeeze(0)


