from model.frontend.pointnet import PointNetEncoder
from model.frontend.pointnet_vlsat import PointNetfeat
from model.backend.gat import BFeatVanillaGAT
from model.frontend.relextractor import *
from model.backend.classifier import consine_classification_obj
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index
import torch
import torch.nn as nn
import torch.nn.functional as F

class BFeatEncoderPretrainNet(BaseNetwork):
    def __init__(self, config, device):
        super(BFeatEncoderPretrainNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=1024)
        self.relation_encoder = RelFeatPointExtractor(config, device, out_dims=1024)
                
    
                
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, # remaining for other processing domain
        is_finetune=True,
    ):
        if is_finetune:
            bsz = obj_pts.shape[0] // 2
            obj_feats_cat, _, _ = self.point_encoder(obj_pts)
            obj_t1_feats = obj_feats_cat[:bsz, ...]
            obj_t2_feats = obj_feats_cat[bsz:, ...]
            
            obj_feats = torch.stack([obj_t1_feats, obj_t2_feats]).mean(dim=0)
            
            edge_feats_cat, _, _ = self.relation_encoder(edge_pts)
            edge_t1_feats = edge_feats_cat[:bsz, ...]
            edge_t2_feats = edge_feats_cat[bsz:, ...]
            
            edge_feats = torch.stack([edge_t1_feats, edge_t2_feats]).mean(dim=0)
            
            return obj_feats, edge_feats, obj_t1_feats, obj_t2_feats, edge_t1_feats, edge_t2_feats
        else:
            obj_feats_cat, _, _ = self.point_encoder(obj_pts)
            edge_feats_cat, _, _ = self.relation_encoder(edge_pts)
            return obj_feats, edge_feats