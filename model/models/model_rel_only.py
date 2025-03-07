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

class BFeatRelObjConNet(BaseNetwork):
    def __init__(self, config, device):
        super(BFeatRelObjConNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts)
        # self.point_encoder.load_state_dict(torch.load(self.t_config.ckp_path))
        # self.point_encoder = self.point_encoder.to(self.device).eval()
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        if self.m_config.relation_type == "pointnet":
            self.relation_encoder = RelFeatPointExtractor(
                config, device
            )
        elif self.m_config.relation_type == "resnet":
            self.relation_encoder = RelFeatNaiveExtractor(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats,
                num_layers=self.m_config.num_layers
            ).to(self.device)
        elif self.m_config.relation_type == "1dconv":
            self.relation_encoder = RelFeatMergeExtractor(
                self.m_config.dim_obj_feats,
                self.m_config.dim_geo_feats,
                self.m_config.dim_edge_feats
            ).to(self.device)
        else:
            raise NotImplementedError
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, # remaining for other processing domain
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        is_train = True
    ):
        if is_train:
            bsz = obj_pts.shape[0] // 2
            obj_feats_cat, _, _ = self.point_encoder(obj_pts)
            obj_t1_feats = obj_feats_cat[:bsz, ...]
            obj_t2_feats = obj_feats_cat[bsz:, ...]
            
            obj_feats = torch.stack([obj_t1_feats, obj_t2_feats]).mean(dim=0)
        else:
            obj_feats, _, _ = self.point_encoder(obj_pts)
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        else:
            edge_feats = self.relation_encoder(edge_pts)
        
        if is_train:
            return obj_feats, edge_feats, obj_t1_feats, obj_t2_feats
        else:
            return obj_feats, edge_feats

class BFeatRelOnlyNet(BaseNetwork):
    def __init__(self, config, obj_gt_feat_mat, device):
        super(BFeatRelOnlyNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        self.obj_gt_feat_mat = obj_gt_feat_mat
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts)
        self.point_encoder.load_state_dict(torch.load(self.t_config.ckp_path))
        self.point_encoder = self.point_encoder.to(self.device).eval()
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.relation_encoder = RelFeatNaiveExtractor(
            self.m_config.dim_obj_feats,
            self.m_config.dim_geo_feats,
            self.m_config.dim_edge_feats,
            self.m_config.num_layers
        ).to(self.device)
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
    ):
        with torch.no_grad():
            _obj_feats, _, _ = self.point_encoder(obj_pts)
        obj_feats = _obj_feats.clone().detach()
        
        x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
        geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
        edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
                
        obj_pred = consine_classification_obj(self.obj_gt_feat_mat, obj_feats)
        
        return edge_feats, obj_pred



class BFeatGNNRelOnlyNet(BaseNetwork):
    def __init__(self, config, device):
        super(BFeatGNNRelOnlyNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts)
        self.point_encoder.load_state_dict(torch.load(self.t_config.ckp_path))
        self.point_encoder = self.point_encoder.to(self.device).eval()
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.relation_encoder = RelFeatNaiveExtractor(
            self.m_config.dim_obj_feats,
            self.m_config.dim_geo_feats,
            self.m_config.dim_edge_feats,
            num_layers=self.m_config.num_layers
        ).to(self.device)
        
        self.gat = BFeatVanillaGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
        
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        batch_ids=None
    ):
        with torch.no_grad():
            _obj_feats = self.point_encoder(obj_pts)
        obj_feats = _obj_feats.clone().detach()
        
        x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
        geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
        edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        
        obj_center = descriptor[:, :3].clone()
        _, edge_gnn_feats = self.gat(
            obj_feats, edge_feats, edge_indices, batch_ids, obj_center
        )
        
        return edge_gnn_feats, obj_feats