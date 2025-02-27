from model.frontend.pointnet import PointNetEncoder
from model.backend.gat import BFeatVanillaGAT
from model.backend.triplet import TripletContrastiveMLPLayer, ProjectHead
from model.frontend.relextractor import *
from model.models.baseline import BaseNetwork
from utils.model_utils import Gen_Index
import torch

class BFeatFullSCLNet(BaseNetwork):
    def __init__(self, config, device):
        super(BFeatFullSCLNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=1024)
        self.mlp_fit_obj = nn.Linear(1024, self.m_config.dim_obj_feats)
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.triplet_encoder = TripletContrastiveMLPLayer(self.m_config.dim_obj_feats, n_layers=6)
        # self.obj_proj = ProjectHead(dims=[ self.m_config.dim_obj_feats, 512, self.m_config.dim_obj_feats ])
        # self.pred_proj = ProjectHead(dims=[ self.m_config.dim_edge_feats, 512, self.m_config.dim_edge_feats ])
        
        self.gat = BFeatVanillaGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            edge_attn=self.m_config.edge_attention,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
        if self.m_config.relation_type == "pointnet":
            self.relation_encoder = RelFeatPointExtractor(
                config, device, out_dims=1024
            )
            self.mlp_fit_rel = nn.Linear(1024, self.m_config.dim_edge_feats)
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
        batch_ids: torch.Tensor,
        is_train = True
    ):
        obj_feats, trans, _ = self.point_encoder(obj_pts)
        obj_feats = self.mlp_fit_obj(obj_feats)
        
        if not self.m_config.relation_type == "pointnet":
            x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
            geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
            edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        else:
            edge_feats = self.relation_encoder(edge_pts)
            edge_feats = self.mlp_fit_rel(edge_feats)
        
        obj_center = descriptor[:, :3].clone()
        obj_gnn_feats, edge_gnn_feats = self.gat(
            obj_feats, edge_feats, edge_indices, batch_ids, obj_center
        )
        sub_tri_feats, obj_tri_feats = self.index_get(obj_gnn_feats, edge_indices)
        tri_feats = self.triplet_encoder(sub_tri_feats, edge_gnn_feats, obj_tri_feats)
        
        # obj_gnn_feats = self.obj_proj(obj_gnn_feats)
        # edge_gnn_feats = self.pred_proj(edge_gnn_feats) # Consider approve this
        
        if is_train:
            return obj_gnn_feats, edge_gnn_feats, tri_feats, trans
        else:
            return obj_gnn_feats, edge_gnn_feats
